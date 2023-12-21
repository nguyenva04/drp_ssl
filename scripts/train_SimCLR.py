import argparse
import torch
import ignite.distributed as idist

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage

from clearml import Task
from drp.twod.drp2d import DRP2D
from ignite.handlers import TerminateOnNan
from ignite.contrib.handlers import (
    ProgressBar,
    create_lr_scheduler_with_warmup,
)

from model.simclr.Encoder import ModifiedResNet
from model.transformation.PairTransform import PairTransform
from ignite.contrib.handlers.clearml_logger import *
from ignite.handlers import Checkpoint
from simclr_stl.simclr.model.Lars import LARS


def training(_, config):

    dataset = DRP2D(root=config.data_root, download=True, train=True, version="1.0", host='islin-hdpmas1', port=5431,
                    transform=PairTransform(dataset=config.dataset_name, type_transform=config.type_transform))

    dataloader = idist.auto_dataloader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True
    )

    model = ModifiedResNet(config.model_name, feature_dim=config.feature_dim)

    model = idist.auto_model(model)

    if config.optimizer == 'Adam':
        optimizer = Adam(
            [params for params in model.parameters() if params.requires_grad],
            lr=config.lr,
            weight_decay=1e-4
        )
    elif config.optimizer == 'LARS':
        optimizer = LARS(
            [params for params in model.parameters() if params.requires_grad],
            lr=config.lr,
            weight_decay=1e-5
        )
    else:
        print("Invalid optimizer")

    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=len(dataloader), eta_min=0)
    lr_scheduler = create_lr_scheduler_with_warmup(
        lr_scheduler=lr_scheduler,
        warmup_start_value=config.lr*0.1,
        warmup_end_value=config.lr,
        warmup_duration=10*len(dataloader)
    )

    temperature = config.temperature

    # ATTENTION en DDP : batch_size doit être scalé
    def loss_fn(out_1, out_2):

        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * config.batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * config.batch_size, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss

    def train_fn(_, batch):
        model.train()
        optimizer.zero_grad()
        (x_i, x_j), _ = batch
        x_i = x_i.to(device=idist.device(), non_blocking=True).float()
        x_j = x_j.to(device=idist.device(), non_blocking=True).float()
        _, z_i = model(x_i)
        _, z_j = model(x_j)
        loss = loss_fn(z_i, z_j)
        loss.backward()
        optimizer.step()
        return loss

    trainer = Engine(train_fn)

    ProgressBar(persist=False).attach(trainer, metric_names="all")
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    @trainer.on(Events.EPOCH_COMPLETED)
    def _():
        epoch = trainer.state.epoch
        max_epochs = trainer.state.max_epochs
        loss = trainer.state.metrics["loss"]
        trainer.logger.info("Train Epoch [{}/{}] : NT-Xent Loss = {}".format(epoch, max_epochs, loss))

    tb_logger = ClearMLLogger(
        project_name="DRP",
        task_name="Training_128_simclr"
    )

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
        ClearMLSaver(dirname='checkpoint202', require_empty=False),
        filename_prefix="checkpoint_101",
        n_saved=1
    )

    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    trainer.run(dataloader, max_epochs=config.max_epochs)


if __name__ == "__main__":

    task = Task.init(
        project_name="DRP",
        task_name="Training_128_simclr"
    )

    parser = argparse.ArgumentParser("DRP-128 SimCLR")
    parser.add_argument("--backend", default="gloo", type=str, help="Backend for distributed computation")
    parser.add_argument("--data_root", default='C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/scripts/data',
                        type=str, help="Path to 2D dataset")
    parser.add_argument("--feature_dim", default=128, type=int, help="Feature dim for latent vector")
    parser.add_argument("--temperature", default=0.1, type=float,
                        help="Temperature used in NT-Xent Loss")
    parser.add_argument("--batch_size", default=256, type=int,
                        help="Number of images in each mini-batch")
    parser.add_argument("--max_epochs", default=30, type=int,
                        help="Number of sweeps over the dataset to train")
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
    parser.add_argument("--model_name", default="resnet18", type=str, help="Base model")
    parser.add_argument("--num_workers", default=2, type=int, help="Number workers")
    parser.add_argument("--optimizer", default='LARS', type=str, help="Optimizer")
    parser.add_argument("--type_transform", default='Cutout', type=str, help="Type Transform")
    parser.add_argument("--dataset_name", default="DRPDataset2D", type=str, help="Type dataset for training simclr")
    args = parser.parse_args()

    with idist.Parallel(backend=args.backend) as parallel:
        parallel.run(training, args)
