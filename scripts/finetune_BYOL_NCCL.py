import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.models import resnet18, resnet50
import torch.nn as nn

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os
from model.linearmodel.linearmodel import LinearModel
from omegaconf import OmegaConf
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from drp.twod.drp2d import DRP2D
from clearml import Task, Dataset


def split_dataset(dataset, data_percentage):
    X = [x for x, _ in dataset]
    labels = [label for _, label in dataset]

    # Calculate the size of the subset (10% of the training data)
    assert 0 < data_percentage <= 1, "data_percentage must be between 0 and 1"
    subset_size = int(data_percentage * len(dataset))
    print("subset_size", subset_size)

    # Use StratifiedShuffleSplit to create a stratified subset of the dataset
    sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=42)
    train_indices, _ = next(sss.split(X, labels))
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    print("Length of train_dataset:", len(train_dataset))
    # Get the labels of the training dataset
    labels = [label for _, label in train_dataset]
    # Create a dictionary to store the number of each class
    class_counts = {}
    for label in labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    # Print the number of each class
    print("Number of each class in train_dataset:")
    for label, count in class_counts.items():
        print(f"Label {label}: {count}")


def training(config):
    cfg = {

        "data": {
            "name": config.dataset,
            "num_classes": config.num_classes,
        },

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

    kwargs = {
        "num_classes": config.num_classes,
        "cifar": True,
        "max_epochs": 3,
        "optimizer": "adam",
        "precision": 16,
        "lars": False,
        "lr": 0.1,
        "exclude_bias_n_norm_lars": False,
        "gpus": "0",
        "weight_decay": 0,
        "extra_optimizer_args": {"momentum": 0.9},
        "scheduler": "step",
        "lr_decay_steps": [60, 80],
        "batch_size": 32,
        "num_workers": 2,
        "pretrained_feature_extractor": config.dir_path
    }

    resnet_dict = {"resnet18": resnet18(pretrained=False),
                        "resnet50": resnet50(pretrained=False),
                        }
    backbone = resnet_dict["resnet50"]

    model_path = Dataset.get(
        dataset_name="resnet50",
        dataset_project="DRP").get_local_copy()

    file_name = "resnet50.pth"
    state_dict = torch.hub.load_state_dict_from_url('http://dummy', model_dir=model_path, file_name=file_name)

    backbone.load_state_dict(state_dict)
    weights = backbone.state_dict()['conv1.weight'].mean(dim=1, keepdim=True)
    # backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    backbone.state_dict()['conv1.weight'] = weights
    backbone.maxpool = nn.Identity()

    # load pretrained feature extractor
    if config.framework == "byol":
        state = torch.load(kwargs["pretrained_feature_extractor"])["state_dict"]

        for k in list(state.keys()):
            if "online_network.backbone" in k: #and "online_network" not in k and "target_network" not in k and "online_predictor" not in k:

                state[k.replace("online_network.backbone.", "")] = state[k]
            del state[k]

    elif config.framework == "nnclr":
        state = torch.load(kwargs["pretrained_feature_extractor"])["state_dict"]
        for k in list(state.keys()):
            if "encoder.backbone." in k:  # and "online_network" not in k and "target_network" not in k and "online_predictor" not in k:

                state[k.replace("encoder.backbone.", "")] = state[k]
            del state[k]

    elif config.framework == "moco":
        state = torch.load(kwargs["pretrained_feature_extractor"])["model"]
        for k in list(state.keys()):
            if "encoder_q" in k:  # and "online_network" not in k and "target_network" not in k and "online_predictor" not in k:

                state[k.replace("encoder_q.", "")] = state[k]
            del state[k]
    if config.load_model:
        for k, v in backbone.named_parameters():
            if k.startswith('conv1'):
                print(k)
                print(v)
        backbone.load_state_dict(state, strict=False)
        for k, v in backbone.named_parameters():
            if k.startswith('conv1'):
                print(k)
                print(v)
    backbone.fc = nn.Identity()

    model = LinearModel(backbone, cfg)
    # for k, v in model.named_parameters():
    #     print(k)

    transform_ = transforms.Compose([
        transforms.ToTensor()
    ])

    if config.dataset == "cifar10":
        # Create a custom subset dataset
        train_dataset = CIFAR10(root=config.dir_dataset,
                                download=True, train=False,
                                transform=transform_)

        # Get the labels (classes) and targets (indices of labels)
        X = [x for x, _ in train_dataset]
        labels = [label for _, label in train_dataset]

        # Calculate the size of the subset (10% of the training data)
        assert 0 < config.data_percentage <= 1, "data_percentage must be between 0 and 1"
        subset_size = int(config.data_percentage * len(train_dataset))
        print("subset_size", subset_size)

        # Use StratifiedShuffleSplit to create a stratified subset of the dataset
        sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=42)
        train_indices, _ = next(sss.split(X, labels))
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        print("Length of train_dataset:", len(train_dataset))
        # Get the labels of the training dataset
        labels = [label for _, label in train_dataset]
        # Create a dictionary to store the number of each class
        class_counts = {}
        for label in labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # Print the number of each class
        print("Number of each class in train_dataset:")
        for label, count in class_counts.items():
            print(f"Label {label}: {count}")
        val_dataset = CIFAR10(root=config.dir_dataset,
                              download=True, train=True, transform=transform_)

    else:
        train_dataset = DRP2D(root=config.dir_dataset,
                              download=True, train=True, version=config.version_dataset, host='islin-hdpmas1', port=5431,
                              transform=transform_)

        val_dataset = DRP2D(root=config.dir_dataset,
                            download=True, train=False, version="1.0", host='islin-hdpmas1', port=5431,
                            transform=transforms.ToTensor())
    train_dl = DataLoader(train_dataset, batch_size=kwargs["batch_size"], shuffle=True, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size=kwargs["batch_size"], shuffle=False, drop_last=True)

    # automatically log our learning rate
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # saves the checkout after every epoch
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc1',
        dirpath=config.dir_checkpoint,
        filename='byol-{epoch}-{val_acc1:.2f}',
        mode='max',
        save_top_k=1
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=config.dir_checkpoint)
    trainer = Trainer(accelerator=config.accelerator,
                      logger=tb_logger,
                      devices=config.devices,
                      max_epochs=config.max_epochs,
                      enable_checkpointing=False,
                      max_steps=-1,
                      callbacks=[lr_monitor],
                      log_every_n_steps=9,
                      # strategy="ddp",
                      )

    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    os.environ['HTTP_PROXY'] = "http://irproxy:8082"
    os.environ['HTTPS_PROXY'] = "http://irproxy:8082"

    os.environ['no_proxy'] = "localhost,islin-hdplnod05,10.68.0.250.nip.io"

    parser = argparse.ArgumentParser(description="PyTorch Implementation of SimCLR")
    parser.add_argument("--dir_dataset",
                        default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/scripts/data",
                        type=str, help="dir to dataset")
    parser.add_argument("--dataset", default="DRPDataset2D", type=str, help="Name of dataset used for training BYOL")
    parser.add_argument("--data_percentage", default=0.1, type=float, help="Percentage of data to train (0-1]")
    parser.add_argument("--max_epochs", default=50, type=int, help="Number of sweeps over the dataset to train")
    parser.add_argument("--load_model", default=True, type=bool, help="turn on load model")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate ")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size for training")
    parser.add_argument("--workers", default=2, type=int, help="Subprocesses to use for data loading")
    parser.add_argument("--num_classes", default=9, type=int, help="Number of classes in the dataset")
    parser.add_argument("--finetune", default=False, type=bool, help="Type Evaluation")
    parser.add_argument("--dir_path",
                        default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/checkpoint/checkpoint_simclr/trained_moco_VerticalFlip, HorizontalFlip, GaussianBlur, Cutout_checkpoint_28897.pt",
                        type=str, help="dir pretrained model")
    parser.add_argument("--devices", default=1, type=int,
                        help="devices")
    parser.add_argument("--framework", default='moco', type=str, help="name of framwork")
    parser.add_argument("--optimizer", default='sgd', type=str, help="Optimizer")
    parser.add_argument("--accelerator", default="auto", type=str, help="Supports passing different accelerator types")
    parser.add_argument("--dir_checkpoint", default="C:/Users/nguyenva/Documents/simclr-stage", type=str,
                        help="dir to save checkpoint")
    parser.add_argument("--version_dataset", default='1.1', type=str, help="Version drpdataset")
    args = parser.parse_args()

    Task.init(project_name="DRP",
              task_name="test_finetune" + args.__getattribute__("framework"))

    training(args)

