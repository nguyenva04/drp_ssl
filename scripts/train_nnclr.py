from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer

from clearml import Task
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from drp.twod.drp2d import DRP2D

from omegaconf import OmegaConf
from model.nnclr.NNCLR import NNCLR
from model.transformation.PairTransform import PairTransform


def training(config):

    cfg = {
        "backbone": {
            "name": "resnet50",

        },
        "data": {
            "name": config.dataset,
            "num_classes": config.num_classes,
            "cifar": config.cifar
        },

        "optimizer": {

            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": 1e-5,
            "max_epochs": config.max_epochs,
        },

        "knn_eval": True,
        "scheduler": {
            "name": "step",
            "warmup_start_lr": config.lr * 0.1,
            "warmup_epochs": 5,
            "lr_decay_steps": [20, 40]
        },

        "method_kwargs": {
            "proj_output_dim": 128,
            "proj_hidden_dim": 2048,
            "pred_hidden_dim": 2048,
            "temperature": 0.1,
            "queue_size": 4096,
        }
    }

    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)

    model = NNCLR(cfg)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor='nnclr_loss',
        dirpath=config.dir_path,
        filename='nnclr-ResNet50',
        mode='min',
        save_top_k=1
    )

    train_dataset = DRP2D(root=config.dir_path,
                          download=True, train=True, version="1.0", host='islin-hdpmas1', port=5431,
                          transform=PairTransform(dataset="DRPDataset2D", type_transform=config.name_transform))
    val_dataset = DRP2D(root=config.dir_path,
                        download=True, train=False, version="1.0", host='islin-hdpmas1', port=5431,
                        transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    trainer = Trainer(accelerator=config.accelerator,
                      devices=config.devices,
                      max_epochs=config.max_epochs,
                      enable_checkpointing=True,
                      max_steps=-1,
                      callbacks=[lr_monitor, checkpoint_callback],
                      log_every_n_steps=9,
                      strategy="ddp",
                      )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    import argparse
    import os
    os.environ['HTTP_PROXY'] = "http://irproxy:8082"
    os.environ['HTTPS_PROXY'] = "http://irproxy:8082"

    os.environ['no_proxy'] = "localhost,islin-hdplnod05,10.68.0.250.nip.io"

    parser = argparse.ArgumentParser(description="PyTorch Implementation of SimCLR")

    parser.add_argument("--max_epochs", default=10, type=int, help="Number of sweeps over the dataset to train")
    parser.add_argument("--lr", default=0.06, type=float, help="Learning rate ")
    parser.add_argument("--dataset", default="DRPD", type=str, help="Name of dataset used for training NNCLR")
    parser.add_argument("--name_transform", default="VerticalFlip, HorizontalFlip, GaussianBlur, Cutout", type=str,
                        help="Name of transform used. Only necessary for DRPDataset")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training")
    parser.add_argument("--cifar", default=False, type=bool, help="Cifar10 is True")
    parser.add_argument("--num_classes", default=9, type=int, help="Number of classes in the dataset")
    parser.add_argument("--dir_path",
                        default="C:/Users/nguyenva/Documents/simclr-stage", type=str, help="dir checkpoint")
    parser.add_argument("--devices", default=1, type=int, help="devices")
    parser.add_argument("--accelerator", default="auto", type=str, help="Supports passing different accelerator types")

    args = parser.parse_args()

    Task.init(project_name="examples-internal",
              task_name="NNCLR_"+args.__getattribute__("dataset")+"_solo_learn")

    training(args)

