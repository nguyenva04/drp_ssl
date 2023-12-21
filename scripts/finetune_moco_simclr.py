import argparse
import torch
from pytorch_lightning import Trainer
import torch.nn as nn
from omegaconf import OmegaConf
from solo.utils.metrics import accuracy_at_k, weighted_mean
from torchvision import transforms

from solo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
from model.moco.moco_v1 import Moco
from typing import Any, Callable, Dict, List, Tuple, Union, Sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18, resnet50, resnet101, resnet152
from drp.twod.drp2d import DRP2D
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from model.simclr.Encoder import ModifiedResNet


class simclr_ds(nn.Module):
    def __init__(self, model, dir_pretrained=None):
        super(simclr_ds, self).__init__()
        self.model = ModifiedResNet(model_name=model)
        if dir_pretrained is not None:
            checkpoint = torch.load(dir_pretrained)
            self.model.load_state_dict(checkpoint["model"], strict=False)

    def forward(self, x):
        _, _, x = self.model(x)
        feature = torch.flatten(x, start_dim=1)
        return feature


class moco_ds(torch.nn.Module):
    def __init__(self, dir_pretrained):
        super().__init__()
        base_model1 = resnet50(pretrained=False)
        base_model2 = resnet50(pretrained=False)

        base_model1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        base_model2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.model = Moco(base_encoder1=base_model1, base_encoder2=base_model2, mlp=True, T=0.1)
        if dir_pretrained is not None:
            checkpoint = torch.load(dir_pretrained)
            for key in checkpoint["model"].keys():
                print(key)

            self.model.load_state_dict(checkpoint["model"], strict=False)

        self.layer = nn.Linear(128, 9)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        _, _, x, _, _, _, _ = self.model(im_q=x, im_k=x)
        # x = self.layer(self.dropout(x))
        x = self.layer(x)

        return x


class FinetuneModel(pl.LightningModule):
    def __init__(self, cfg):
        super(FinetuneModel, self).__init__()
        if cfg.model == "simclr":
            self.backbone = simclr_ds(model="resnet50", dir_pretrained=cfg.pretrained_feature_extractor)
            self.classifier = nn.Sequential(nn.Linear(2048, 256),
                                            nn.Linear(256, 128),
                                            nn.Linear(128, 9))
        elif cfg.model == "moco":
            self.backbone = moco_ds(cfg.pretrained_feature_extractor)
            self.classifier = nn.Linear(128, 9)
        self.finetune = cfg.finetune
        self.lr = cfg.optimizer.lr
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.scheduler: str = cfg.scheduler.name
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.max_epochs: int = cfg.max_epochs
        self.warmup_start_lr: int = cfg.scheduler.warmup_start_lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.num_classes: int = cfg.data.num_classes
        self.validation_step_outputs = []

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """
        if self.finetune:

            return [
                {"name": "backbone", "params": self.backbone.parameters()},
                {"name": "classifier", "params": self.classifier.parameters()},
            ]
        else:
            return [
                {"name": "classifier", "params": self.classifier.parameters()},
            ]

    def configure_optimizers(self) -> Tuple[List, List]:
        learnable_params = self.learnable_params
        # learnable_params = list(filter(lambda p: p.requires_grad, self.parameters()))

        # optimizer = LARS(learnable_params, lr=self.lr, momentum=self.weight_decay)
        optimizer = torch.optim.SGD(learnable_params,
                                    self.lr,
                                    momentum=0.9,
                                    weight_decay=1.5e-6)
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

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        feats = self.backbone(X)

        logits = self.classifier(feats)

        return {"logits": logits, "feats": feats}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:

        X, target = batch
        out = self(X)
        loss = F.cross_entropy(out["logits"], target)
        acc1, acc5 = accuracy_at_k(out["logits"], target, top_k=(1, 5))

        log = {"train_loss": loss, "train_acc1": acc1}
        self.log_dict(log, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        X, target = batch

        out = self(X)
        loss = F.cross_entropy(out["logits"], target)
        _, predicted = torch.max(out["logits"], 1)
        acc1, acc5 = accuracy_at_k(out["logits"], target, top_k=(1, 5))
        metrics = {
            "batch_size": X.size(0),
            "val_loss": loss,
            "val_acc1": acc1,
            "out": predicted.detach(),
            "feats": out["feats"].detach(),
            "target": target.detach()
        }
        self.validation_step_outputs.append(metrics)
        return metrics

    def on_validation_epoch_end(self):
        feature = torch.cat([item["out"] for item in self.validation_step_outputs], dim=0)
        feature_ = torch.cat([item["feats"] for item in self.validation_step_outputs], dim=0)
        target = torch.cat([item["target"] for item in self.validation_step_outputs], dim=0)

        # Convert the feature tensor & the target tensor to a numpy array
        feature_np = feature.cpu().numpy()
        feature_np_ = feature_.cpu().numpy()
        target_np = target.cpu().numpy()

        # Calculate evaluation metrics
        # accuracy = accuracy_score(target_np, feature_np)
        precision = precision_score(target_np, feature_np, average='weighted', zero_division=0)
        recall = recall_score(target_np, feature_np, average='weighted', zero_division=0)
        f1 = f1_score(target_np, feature_np, average='weighted', zero_division=0)

        val_loss = weighted_mean(self.validation_step_outputs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(self.validation_step_outputs, "val_acc1", "batch_size")

        # Perform t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, learning_rate='auto', init='random')
        # feature_np = PCA(n_components=50).fit_transform(feature_np)
        tsne_result = tsne.fit_transform(feature_np_)

        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=target_np, s=35,
                    cmap=plt.cm.get_cmap("jet", self.num_classes))

        plt.colorbar(label='Target')
        plt.title('t-SNE Visualization')

        self.loggers[0].experiment.add_figure("t-SNE Visualization on cifar10 test set", plt.gcf(), self.current_epoch)
        # Confusion matrix
        cm = confusion_matrix(target_np, feature_np)
        plt.figure(figsize=(10, 10))
        # pl.matshow(cm)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.annotate(str(cm[i, j]), xy=(j, i), ha='center', va='center')
        classes = ['0', '1', '2', '3', '4',
                   '5', '6', '7', '8']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        plt.xlabel('Predicted')
        plt.ylabel('True')
        # plt.show()
        self.loggers[0].experiment.add_figure("Confusion Matrix", plt.gcf(), self.current_epoch)
        self.validation_step_outputs.clear()

        log = {"val_loss": val_loss, "val_acc1": val_acc1,  # "val_acc5": val_acc5,
               "precision": precision, "recall": recall, "f1": f1}

        self.log_dict(log, sync_dist=True)


def training(config):
    cfg = {

        "data": {
            "num_classes": config.num_classes,
        },
        "model": config.model,

        "optimizer": {
            "name": config.optimizer,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": 1e-5,
            # "kwargs": {"momentum": 0.9},
        },
        "max_epochs": config.max_epochs,

        "scheduler": {
            "name": "none",
            "warmup_start_lr": config.lr * 0.1,
            "min_lr": 0.001,
            "interval": "step",
            "warmup_epochs": 5,
            "lr_decay_steps": [20, 40]
        },

        "finetune": config.finetune,
        "performance": {"disable_channel_last": False},
        "pretrained_feature_extractor": config.dir_path,

    }

    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)

    model = FinetuneModel(cfg)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc1',
        dirpath=config.dir_checkpoint,
        filename=config.model + "-{epoch}-{val_acc1:.2f}",
        mode='max',
        save_top_k=1
    )
    train_dataset = DRP2D(root=config.dir_dataset,
                          download=True, train=True, version=config.version_dataset, host='islin-hdpmas1', port=5431,
                          transform=transforms.ToTensor())

    val_dataset = DRP2D(root=config.dir_dataset,
                        download=True, train=False, version="1.0", host='islin-hdpmas1', port=5431,
                        transform=transforms.ToTensor())

    train_dl = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir=config.dir_path)
    trainer = Trainer(accelerator=config.accelerator,
                      # logger=tb_logger,
                      devices=config.devices,
                      max_epochs=config.max_epochs,
                      enable_checkpointing=False,
                      max_steps=-1,
                      callbacks=[lr_monitor],
                      log_every_n_steps=9,
                      strategy="ddp",
                      )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    import argparse
    import os
    from clearml import Task, Dataset

    os.environ['HTTP_PROXY'] = "http://irproxy:8082"
    os.environ['HTTPS_PROXY'] = "http://irproxy:8082"

    os.environ['no_proxy'] = "localhost,islin-hdplnod05,10.68.0.250.nip.io"

    parser = argparse.ArgumentParser(description="PyTorch Implementation of SimCLR")
    parser.add_argument("--dir_dataset",
                        default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/scripts/data",
                        type=str, help="dir to dataset")
    parser.add_argument("--max_epochs", default=50, type=int, help="Number of sweeps over the dataset to train")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate ")
    parser.add_argument("--model", default="simclr", type=str, help="model name: simclr or moco")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training")
    parser.add_argument("--workers", default=2, type=int, help="Subprocesses to use for data loading")
    parser.add_argument("--num_classes", default=9, type=int, help="Number of classes in the dataset")
    parser.add_argument("--finetune", default=False, type=bool, help="Type Evaluation")
    parser.add_argument("--dir_path",
                        default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/checkpoint/checkpoint_simclr/trained_simclr_checkpoint_5000.pt",
                        type=str, help="dir pretrained model")
    parser.add_argument("--devices", default=1, type=int,
                        help="devices")
    parser.add_argument("--optimizer", default='sgd', type=str, help="Optimizer")
    parser.add_argument("--accelerator", default="auto", type=str, help="Supports passing different accelerator types")
    parser.add_argument("--dir_checkpoint", default="C:/Users/nguyenva/Documents/simclr-stage", type=str,
                        help="dir to save checkpoint")
    parser.add_argument("--version_dataset", default='1.1', type=str, help="Version drpdataset")
    args = parser.parse_args()

    Task.init(project_name="examples-internal",
              task_name=args.__getattribute__("model")+"finetune")

    training(args)

