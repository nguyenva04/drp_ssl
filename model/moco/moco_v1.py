import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50, resnet101, resnet152


class Moco(nn.Module):
    def __init__(self, base_encoder1, base_encoder2, dim=128, K=8192, m=0.999, T=0.1, mlp=False):
        super(Moco, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dim = dim

        self.encoder_q = base_encoder1
        in_features = self.encoder_q.fc.in_features
        self.encoder_q.fc = nn.Linear(in_features, self.dim)
        # self.projection_q = nn.Sequential(nn.Linear(self.dim, in_features), nn.ReLU(), nn.Linear(in_features, in_features), nn.ReLU(), nn.Linear(in_features, self.dim))
        self.encoder_k = base_encoder2
        self.encoder_k.fc = nn.Linear(in_features, self.dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = in_features
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),  self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),  self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for parameter_q, parameter_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            parameter_k.data.copy_(parameter_k.data * self.m + parameter_q.data * (1.0 - self.m))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q_i = self.encoder_q(im_q)  # queries: NxC
        # q_j = self.encoder_q(im_k)
        # q_1 = self.projection_q(q_i)
        # q_2 = self.projection_q(q_j)
        q = nn.functional.normalize(q_i, dim=1)


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            # k_1 = self.encoder_k(im_q)  # keys: NxC
            k_2 = self.encoder_k(im_k)  # keys: NxC

            k = nn.functional.normalize(k_2, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, q_i#, q_1, q_2, k_1, k_2


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == "__main__":
    model = resnet50(num_classes=128)
    dim_mlp = model.fc.weight.shape[1]
    model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
    print(model)

