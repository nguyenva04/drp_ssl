import torch

import omegaconf
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pytorch_lightning as pl

from clearml import Dataset
from torchvision.models.resnet import resnet50, resnet18, resnet101, resnet152
from typing import Any, Dict, List, Tuple, Union, Sequence

from solo.utils.lars import LARS
from model.outil.knn import KMeansClustering
from sklearn.manifold import TSNE

from torch.optim.lr_scheduler import MultiStepLR
from solo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR


def InfoNCE_loss(nearest_point, positive_point, temperature=1.0, reduction="mean"):
    if nearest_point.dim() != 2:
        raise ValueError('<nearest_point> must have 2 dimensions.')
    if positive_point.dim() != 2:
        raise ValueError('<positive_point> must have 2 dimensions.')
    nn = F.normalize(nearest_point, p=2, dim=1)
    pred = F.normalize(positive_point, p=2, dim=1)

    batch_size, _ = nn.shape
    labels = torch.arange(batch_size).to(nn.device)

    logits = (nn @ pred.T) / temperature

    return F.cross_entropy(logits, labels, reduction=reduction)


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
            nn.BatchNorm1d(projection_size),
        )

    def forward(self, x):
        return self.net(x)


class BaseMethod(nn.Module):
    def __init__(self, backbone_name, proj_hidden_dim, proj_output_dim, cifar=True):
        """
        Cfg basic structure:
            backbone:
                name (str): architecture of the base backbone.
                kwargs (dict): extra backbone kwargs.
        """
        super(BaseMethod, self).__init__()

        self.resnet_dict = {"resnet18": resnet18(pretrained=False),
                            "resnet50": resnet50(pretrained=False),
                            "resnet101": resnet101(pretrained=False),
                            "resnet152": resnet152(pretrained=False),
                            }
        self.backbone = self.resnet_dict[backbone_name]

        model_path = Dataset.get(
            dataset_name=backbone_name,
            dataset_project="DRP").get_local_copy()

        file_name = f"{backbone_name}.pth"
        state_dict = torch.hub.load_state_dict_from_url('http://dummy', model_dir=model_path, file_name=file_name)

        self.features_dim: int = self.backbone.fc.in_features

        self.backbone.load_state_dict(state_dict)
        if cifar:
            self.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            weights = self.backbone.state_dict()['conv1.weight'].mean(dim=1, keepdim=True)
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.backbone.state_dict()['conv1.weight'] = weights

        self.backbone.maxpool = torch.nn.Identity()
        self.backbone.fc = nn.Identity()

        # projector
        self.projector = MLP(dim=self.features_dim, projection_size=proj_output_dim, hidden_size=proj_hidden_dim)

    def forward(self, x, return_embedding=False):
        embedding = self.backbone(x)
        embedding = torch.flatten(embedding, start_dim=1)
        if return_embedding:
            return embedding

        return self.projector(embedding)


class NNCLR(pl.LightningModule):
    def __init__(self, cfg):
        super(NNCLR, self).__init__()

        self.cfg: omegaconf.DictConfig = cfg

        self.lr: float = cfg.optimizer.lr
        self.backbone_name: str = cfg.backbone.name

        self.num_classes: int = cfg.data.num_classes
        self.weight_decay: float = cfg.optimizer.weight_decay

        self.scheduler: str = cfg.scheduler.name
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.max_epochs: int = cfg.optimizer.max_epochs
        self.warmup_start_lr: int = cfg.scheduler.warmup_start_lr

        self.temperature: float = cfg.method_kwargs.temperature
        self.queue_size: int = cfg.method_kwargs.queue_size

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim

        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.knn_eval: bool = cfg.knn_eval

        self.encoder: nn.Module = BaseMethod(backbone_name=self.backbone_name,
                                             proj_hidden_dim=proj_hidden_dim,
                                             proj_output_dim=proj_output_dim,
                                             cifar=cfg.data.cifar
                                             )

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        if self.knn_eval:
            self.kmeans = KMeansClustering(n_clusters=self.num_classes)
        # queue
        self.register_buffer("queue", torch.randn(self.queue_size, proj_output_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.validation_step_outputs = []

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(NNCLR, NNCLR).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")
        return cfg

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "encoder", "params": self.encoder.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()}
        ]

    def configure_optimizers(self) -> Tuple[List, List]:

        optimizer = LARS(self.learnable_params, lr=self.lr, momentum=self.weight_decay)
        # optimizer = torch.optim.SGD(self.learnable_params,
        #                             self.lr,
        #                             momentum=0.9,
        #                             weight_decay=1.5e-6)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.scheduler.lower() == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":

            scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,
                                                      warmup_epochs=self.warmup_epochs,
                                                      max_epochs=self.max_epochs,
                                                      warmup_start_lr=self.warmup_start_lr)
        elif self.scheduler == "step":

            scheduler = MultiStepLR(optimizer, milestones=self.lr_decay_steps, gamma=0.1)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")
        return [optimizer], [scheduler]

    @torch.no_grad()
    def dequeue_and_enqueue(self, z: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner. Also stores
        the labels of the samples.

        Args:
            z (torch.Tensor): batch of projected features.
            y (torch.Tensor): labels of the samples in the batch.
        """

        # z = self.encoder(z)

        batch_size = z.shape[0]

        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0

        self.queue[ptr: ptr + batch_size, :] = z
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr  # type: ignore

    @torch.no_grad()
    def find_nn(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds the nearest neighbor of a sample.

        Args:
            z (torch.Tensor): a batch of projected features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                indices and projected features of the nearest neighbors.
        """

        idx = (z @ self.queue.T).max(dim=1)[1]
        nn = self.queue[idx]
        return nn

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> Dict[str, Any]:

        z_i, z_j = self.encoder(x_i), self.encoder(x_j)
        p_i, p_j = self.predictor(z_i), self.predictor(z_j)

        return {"z_i": z_i, "z_j": z_j, "p_i": p_i, "p_j": p_j}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:

        (x_i, x_j), _ = batch

        out = self(x_i, x_j)
        # find nn
        nn_i = self.find_nn(out["z_i"])
        nn_j = self.find_nn(out["z_j"])

        # ------- contrastive loss -------
        nnclr_loss = (
                InfoNCE_loss(nn_i, out["p_j"], temperature=self.temperature)
                + InfoNCE_loss(nn_j, out["p_i"], temperature=self.temperature)
        )
        loss = nnclr_loss.mean()

        # dequeue and enqueue
        self.dequeue_and_enqueue(out["z_i"])
        metrics = {
            "nnclr_loss": loss
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:

        X, target = batch
        out = self(X, X)

        output_steps = {
            "feats": out["z_i"].detach(),
            "target": target.detach()
        }
        self.validation_step_outputs.append(output_steps)

    def on_validation_epoch_end(self) -> None:

        feature = torch.cat([item["feats"] for item in self.validation_step_outputs], dim=0)
        target = torch.cat([item["target"] for item in self.validation_step_outputs], dim=0)

        # Convert the feature tensor & the target tensor to a numpy array
        feature_np = feature.cpu().numpy()
        target_np = target.cpu().numpy()

        acc = self.kmeans.fit(feature)
        log = {"silhouette_score": acc}

        self.log_dict(log, sync_dist=True)

        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, learning_rate='auto', init='random')
        # feature_np = PCA(n_components=50).fit_transform(feature_np)
        tsne_result = tsne.fit_transform(feature_np)

        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=target_np, s=35, cmap=plt.cm.get_cmap("jet", self.num_classes))

        plt.colorbar(label='Target')
        plt.title('t-SNE Visualization')

        self.loggers[0].experiment.add_figure("t-SNE Visualization on cifar10 test set", plt.gcf(), self.current_epoch)
        self.validation_step_outputs.clear()
