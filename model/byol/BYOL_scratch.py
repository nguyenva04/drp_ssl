import math
import torch

import omegaconf
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from clearml import Task, Dataset
from torchvision.models.resnet import resnet50, resnet18, resnet101, resnet152
from typing import Any, Dict, List, Tuple, Union, Sequence

from solo.utils.lars import LARS
from model.outil.knn import KMeansClustering
from sklearn.manifold import TSNE

from torch.optim.lr_scheduler import MultiStepLR

from solo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from omegaconf import OmegaConf


@torch.no_grad()
def initialize_momentum_params(online_net: nn.Module, momentum_net: nn.Module):
    """Copies the parameters of the online network to the momentum network.

    Args:
        online_net (nn.Module): online network (e.g. online backbone, online projection, etc...).
        momentum_net (nn.Module): momentum network (e.g. momentum backbone,
            momentum projection, etc...).
    """

    params_online = online_net.parameters()
    params_momentum = momentum_net.parameters()
    for po, pm in zip(params_online, params_momentum):
        pm.data.copy_(po.data)
        pm.requires_grad = False


class MomentumUpdater:
    def __init__(self, base_tau: float = 0.996, final_tau: float = 1.0):
        """Updates momentum parameters using exponential moving average.

        Args:
            base_tau (float, optional): base value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 0.996.
            final_tau (float, optional): final value of the weight decrease coefficient
                (should be in [0,1]). Defaults to 1.0.
        """

        super().__init__()

        assert 0 <= base_tau <= 1
        assert 0 <= final_tau <= 1 and base_tau <= final_tau

        self.base_tau = base_tau
        self.cur_tau = base_tau
        self.final_tau = final_tau

    @torch.no_grad()
    def update(self, online_net: nn.Module, momentum_net: nn.Module):
        """Performs the momentum update for each param group.

        Args:
            online_net (nn.Module): online network (e.g. online backbone, online projection, etc...).
            momentum_net (nn.Module): momentum network (e.g. momentum backbone,
                momentum projection, etc...).
        """
        for op, mp in zip(online_net.parameters(), momentum_net.parameters()):
            mp.data = self.cur_tau * mp.data + (1 - self.cur_tau) * op.data

    def update_tau(self, cur_step: int, max_steps: int):
        """Computes the next value for the weighting decrease coefficient tau using cosine annealing.

        Args:
            cur_step (int): number of gradient steps so far.
            max_steps (int): overall number of gradient steps in the whole training.
        """

        self.cur_tau = (
            self.final_tau
            - (self.final_tau - self.base_tau) * (math.cos(math.pi * cur_step / max_steps) + 1) / 2
        )


def loss_fn(x, y):
    # L2 normalization
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    return 2 - 2 * (x * y).sum(dim=-1)


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
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


class BYOL(pl.LightningModule):
    def __init__(self, cfg):
        super(BYOL, self).__init__()

        self.cfg: omegaconf.DictConfig = cfg

        self.lr: float = cfg.optimizer.lr
        self.backbone_name: str = cfg.backbone.name

        self.num_classes: int = cfg.data.num_classes
        self.weight_decay: float = cfg.optimizer.weight_decay

        self.scheduler: str = cfg.scheduler.name
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.max_epochs: int = cfg.optimizer.max_epochs
        self.warmup_start_lr: int = cfg.scheduler.warmup_start_lr

        self.base_tau: float = 0.99
        self.final_tau: float = 1.0

        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.knn_eval: bool = cfg.knn_eval

        self.momentum_updater = MomentumUpdater(cfg.momentum.base_tau, cfg.momentum.final_tau)

        self.online_network: nn.Module = BaseMethod(backbone_name=self.backbone_name,
                                                    proj_hidden_dim=cfg.method_kwargs.proj_hidden_dim,
                                                    proj_output_dim=cfg.method_kwargs.proj_output_dim,
                                                    cifar=cfg.data.cifar
                                                    )

        # Initialize target networks with online network weights
        self.target_network: nn.Module = BaseMethod(backbone_name=self.backbone_name,
                                                    proj_hidden_dim=cfg.method_kwargs.proj_hidden_dim,
                                                    proj_output_dim=cfg.method_kwargs.proj_output_dim,
                                                    cifar=cfg.data.cifar
                                                    )
        initialize_momentum_params(self.online_network, self.target_network)

        # predictor
        self.online_predictor = MLP(dim=cfg.method_kwargs.proj_output_dim,
                                    projection_size=cfg.method_kwargs.proj_output_dim,
                                    hidden_size=cfg.method_kwargs.pred_hidden_dim)

        if self.knn_eval:
            self.kmeans = KMeansClustering(n_clusters=self.num_classes)

        self.validation_step_outputs = []
        self.min_loss = 1000.0

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "online_network", "params": self.online_network.parameters()},
            {"name": "online_predictor", "params": self.online_predictor.parameters()}
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        learnable_params = self.learnable_params
        # learnable_params = list(filter(lambda p: p.requires_grad, self.parameters()))

        optimizer = LARS(learnable_params, lr=self.lr, momentum=self.weight_decay)
        # optimizer = torch.optim.SGD(learnable_params,
        #                             self.lr,
        #                             momentum=0.9,
        #                             weight_decay=0.0001)
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

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, Any]:
        X, Y = X.float(), Y.float()

        proj_online_1 = self.online_network(X)
        proj_online_2 = self.online_network(Y)

        # additional online's MLP head called predictor
        pred_online_1 = self.online_predictor(proj_online_1)
        pred_online_2 = self.online_predictor(proj_online_2)
        with torch.no_grad():
            # teacher processes the images and makes projections: backbone + MLP

            proj_target_1 = self.target_network(X).detach()
            proj_target_2 = self.target_network(Y).detach()

        return {"pred_online_1": pred_online_1, "pred_online_2": pred_online_2,
                "proj_target_1": proj_target_1, "proj_target_2": proj_target_2}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:

        (x_i, x_j), _ = batch

        out = self(x_i, x_j)

        loss = loss_fn(out["pred_online_2"], out["proj_target_1"]) +\
               loss_fn(out["pred_online_1"], out["proj_target_2"])

        loss = loss.mean()

        metrics = {
            "train_neg_cos_sim": loss
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""

        self.last_step = 0

    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int):
        if self.trainer.global_step > self.last_step:

            self.momentum_updater.update(self.online_network, self.target_network)

            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)

            # update tau
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step,
                max_steps=self.trainer.estimated_stepping_batches,
            )
        self.last_step = self.trainer.global_step

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Dict[str, Any]:

        X, target = batch
        # X = [X, X]
        out = self(X, X)

        output_steps = {
            "feat_1": out["pred_online_1"].detach(),
            "target": target.detach()
        }
        self.validation_step_outputs.append(output_steps)

    def on_validation_epoch_end(self) -> None:

        feature = torch.cat([item["feat_1"] for item in self.validation_step_outputs], dim=0)
        target = torch.cat([item["target"] for item in self.validation_step_outputs], dim=0)

        # Convert the feature tensor & the target tensor to a numpy array
        feature_np = feature.cpu().numpy()
        target_np = target.cpu().numpy()

        # acc = self.kmeans.fit(feature)
        # log = {"silhouette_score": acc}

        # self.log_dict(log, sync_dist=True)

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








