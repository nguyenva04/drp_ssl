import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import ignite.distributed as idist

from ignite.engine import Engine, Events
from ignite.handlers import TerminateOnNan, Checkpoint
from ignite.metrics import RunningAverage

from ignite.utils import setup_logger
from ignite.contrib.handlers.clearml_logger import *
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.handlers import (
    ProgressBar,
    create_lr_scheduler_with_warmup,
)
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torchvision.models.resnet import resnet18, resnet50, resnet101, resnet152
from simclr_stl.simclr.data.load_dataset import load_data

from model.transformation.PairTransform import PairTransform
from drp.twod.drp2d import DRP2D
from model.moco.moco_v1 import Moco
from clearml import Dataset


def loss_fn(x, y):
    # L2 normalization
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    return 2 - 2 * (x * y).sum(dim=-1)


def training(_, config):

    logger = setup_logger(name="Training Moco with" + config.dataset_name)
    logger.info("config = {}".format(config))

    if config.dataset_name == "cifar10":
        dataset = load_data(config.root_data, dataset_name=config.dataset_name, transform=PairTransform(dataset=config.dataset_name))
    elif config.dataset_name == "DRPDataset2D":
        dataset = DRP2D(root=config.root_data, download=True, train=True, version="1.0", host='islin-hdpmas1', port=5431,
                        transform=PairTransform(dataset="DRPDataset2D", type_transform=config.transform_name))
    else:
        assert config.dataset_name in ['stl10', "cifar10", 'DRPDataset2D'], (f'dataset "{config.dataset_name}" '
                                                                             f'is not supported. Should be "stl10" '
                                                                             f'or "DRPDataset2D".')
    data_loader = idist.auto_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.workers,
        shuffle=True,
        drop_last=True
    )

    base_model1 = resnet50(pretrained=False)
    base_model2 = resnet50(pretrained=False)

    model_path = Dataset.get(dataset_name="resnet50", dataset_project="DRP").get_local_copy()

    state_dict = torch.hub.load_state_dict_from_url(
        'http://dummy', model_dir=model_path, file_name="resnet50.pth")

    base_model1.load_state_dict(state_dict)
    base_model2.load_state_dict(state_dict)

    weights = base_model1.state_dict()['conv1.weight'].mean(dim=1, keepdim=True)
    base_model1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    base_model2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    base_model1.state_dict()['conv1.weight'] = weights
    base_model2.state_dict()['conv1.weight'] = weights

    model = Moco(base_encoder1=base_model1, base_encoder2=base_model2, mlp=True, T=config.temperature)
    model = idist.auto_model(model)

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                momentum=0.9,
                                weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    lr_scheduler = LRScheduler(scheduler)

    def train_fn(_, batch):
        model.train()
        optimizer.zero_grad()
        (x_i, x_j), _ = batch
        x_i = x_i.to(device=idist.device(), non_blocking=True, dtype=torch.float)
        x_j = x_j.to(device=idist.device(), non_blocking=True, dtype=torch.float)
        output, target, _ = model(im_q=x_i, im_k=x_j)
        loss = criterion(output, target)
        # l2 = (loss_fn(q_1, k_2) + loss_fn(q_2, k_1)).mean()
        # loss = (1 - config.lamda)*l1 + config.lamda*l2
        loss.backward()
        optimizer.step()
        return loss

    trainer = Engine(train_fn)
    trainer.logger = logger

    ProgressBar(persist=False).attach(trainer, metric_names="all")

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    @trainer.on(Events.EPOCH_COMPLETED)
    def _():
        epoch = trainer.state.epoch
        max_epochs = trainer.state.max_epochs
        loss = trainer.state.metrics["loss"]
        trainer.logger.info("Train Epoch [{}/{}] : NT-Xent Loss = {}".format(epoch, max_epochs, loss))

    tb_logger = TensorboardLogger(log_dir=config.dir_output + "/tensorboard")
    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(tag="training", metric_names=["loss"]),
        event_name=Events.ITERATION_COMPLETED,
    )

    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer, param_name="lr"),
        event_name=Events.ITERATION_STARTED,
    )

    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
    }

    checkpoint = Checkpoint(
        objects_to_checkpoint,
        ClearMLSaver(dirname=config.dir_output + '/checkpoint/checkpoint_simclr', require_empty=False),
        filename_prefix="trained_moco_"+config.transform_name,
        n_saved=1
    )

    trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    trainer.run(data_loader, max_epochs=config.max_epochs)


if __name__ == "__main__":
    import os
    from clearml import Task

    os.environ['HTTP_PROXY'] = "http://irproxy:8082"
    os.environ['HTTPS_PROXY'] = "http://irproxy:8082"

    os.environ['no_proxy'] = "localhost,islin-hdplnod05,10.68.0.250.nip.io"

    parser = argparse.ArgumentParser(description="PyTorch Implementation of SimCLR")
    parser.add_argument("--backend", default="gloo", type=str, help="Backend for distributed computation")

    parser.add_argument("--dataset_name", default="DRPDataset2D", type=str, help="Type dataset for training simclr")
    parser.add_argument("--root_data",
                        default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/scripts/data",
                        type=str, help="root to dataset")
    parser.add_argument("--transform_name", default="VerticalFlip, HorizontalFlip, GaussianBlur, Cutout", type=str, help="Data Augmentation for training DRP")
    parser.add_argument("--dir_output", default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage",
                        type=str, help="directory of output which includes checkpoint and tensorboard")
    parser.add_argument("--split", default="test", type=str, help="Feature dim for latent vector")
    parser.add_argument("--feature_dim", default=128, type=int, help="Feature dim for latent vector")
    parser.add_argument("--model", default="resnet50", type=str, help="Model architecture")

    parser.add_argument("--temperature", default=0.3, type=float, help="Temperature used in NT-Xent Loss")
    parser.add_argument("--batch_size", default=64, type=int, help="Number of images in each mini-batch")
    parser.add_argument("--max_epochs", default=30, type=int, help="Number of sweeps over the dataset to train")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate ")
    parser.add_argument("--lamda", default=0.1, type=float, help="Weighted loss ")
    parser.add_argument("--workers", default=2, type=int, help="Subprocesses to use for data loading")

    args = parser.parse_args()

    task = Task.init(
        project_name="DRP",
        task_name="Training Moco on DRPDataset"
    )
    with idist.Parallel(backend=args.backend) as parallel:
        parallel.run(training, args)
