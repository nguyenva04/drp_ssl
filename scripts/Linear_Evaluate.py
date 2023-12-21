import argparse

import ignite.distributed as idist
import torch
import torch.nn as nn

from ignite.contrib.handlers.clearml_logger import *
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, create_supervised_evaluator, Engine, create_supervised_trainer
from ignite.handlers import Checkpoint
from ignite.metrics import Accuracy, Loss, RunningAverage, TopKCategoricalAccuracy
from ignite.utils import setup_logger
from torch.optim import SGD, Adam
from torchvision import transforms

from simclr_stl.simclr.data.load_dataset import load_data
from simclr_stl.simclr.model.Encoder import BaseEncoder
from simclr_stl.simclr.model.downstream_model import DSModel
from drp.twod.drp2d import DRP2D
from ignite.contrib.handlers import (
    ProgressBar
    )



def training(_, config):

    logger = setup_logger(name="Retraining SimCLR on " + config.dataset)
    logger.info("config={}".format(config))

    transform = transforms.Compose([
        # transforms.ToPILImage(),
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.4899],
        #                      [0.2899])
    ])
    if config.dataset == "cifar10":
        dataset_train = load_data(data_root='C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/scripts/data',
                            dataset_name="cifar10", transform=transform, train=True)
        dataset_test = load_data(data_root='C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/scripts/data',
                            dataset_name="cifar10", transform=transform, train=False)
    if config.dataset == 'stl10':
        dataset_train = load_data(config.data_root, dataset_name="stl10", split="train", transform=transform)
        dataset_test = load_data(config.data_root, dataset_name="stl10", split="test", transform=transform)
    if config.dataset == 'DRPDataset2D':
        dataset_train = DRP2D(root=config.root_data, download=True, train=True, version=config.version_dataset, host='islin-hdpmas1',
                              port=5431, transform=transform)
        dataset_test = DRP2D(root=config.root_data, download=True, train=False, version="1.0", host='islin-hdpmas1',
                             port=5431, transform=transform)

    train_loader = idist.auto_dataloader(dataset_train,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_worker,
                                         shuffle=True, drop_last=True)
    test_loader = idist.auto_dataloader(dataset_test,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_worker,
                                        shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_encoder = BaseEncoder(model=config.model_name, feature_dim=128)
    model = DSModel(pre_model=base_encoder, nb_classes=config.nb_classes)
    checkpoint = torch.load(config.dir_fturn)
    model.load_state_dict(checkpoint["model"])
    model = idist.auto_model(model)
    for name, param in model.named_parameters():
        param.requires_grad = True
        print(name)

    optimizer_dict = {
        'Adam': Adam(params=model.parameters(),
                     lr=config.lr,
                     weight_decay=1e-4),
        'SGD': SGD(params=model.parameters(),
                   lr=config.lr,
                   momentum=0.9)
    }
    optimizer = optimizer_dict[config.optimizer_name]
    criterion = nn.CrossEntropyLoss().cuda()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dict[config.optimizer_name], step_size=20, gamma=0.1)
    lr_scheduler = LRScheduler(scheduler)


    metrics = {
        "Top1_Accuracy": Accuracy(),
        "Loss": Loss(criterion),
        "Top5_Accuracy": TopKCategoricalAccuracy(k=5)
    }


    def train_step(_, batch):
        x, y = batch
        x = x.to(device=idist.device(), non_blocking=True)
        y = y.to(device=idist.device(), non_blocking=True)

        model.train()
        y_pred = model(x)

        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    # Define a trainer engine
    # trainer = Engine(train_step)

    # creating trainer,evaluator
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    test_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)


    # test_evaluator = Engine(validation_step)
    # tb_logger = ClearMLLogger(
    #     project_name="DRP",
    #     task_name="Linear-Evaluation-Simclr"
    # )
    tb_logger = TensorboardLogger(log_dir=config.dir_output + "/tensorboard/tensorboad_DS")

    # Add progress bar to monitor model training
    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"Batch Loss": x})
    RunningAverage(output_transform=lambda x: x).attach(trainer, "Loss")

    @trainer.on(Events.STARTED)
    def start_message():
        print("Start training!")

    @trainer.on(Events.EPOCH_COMPLETED)
    def _():
        epoch = trainer.state.epoch
        max_epochs = trainer.state.max_epochs
        loss = trainer.state.metrics["Loss"]
        trainer.logger.info("Train Epoch [{}/{}]: Loss = {}".format(epoch, max_epochs, loss))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results():
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        print(
            f"Training Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['Top1_Accuracy']:.3f} "
            f"Top5-Acc: {metrics['Top5_Accuracy']:.3f} Avg loss: {metrics['Loss']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results():
        test_evaluator.run(test_loader)
        metrics = test_evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['Top1_Accuracy']:.3f} "
            f"Top5-Acc: {metrics['Top5_Accuracy']:.3f} Avg loss: {metrics['Loss']:.2f}")

    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer_dict[config.optimizer_name], param_name="lr"),
        event_name=Events.ITERATION_STARTED,
    )

    for tag, evaluator in [("training", train_evaluator), ("testing", test_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator, event_name=Events.EPOCH_COMPLETED,
            tag=tag, metric_names=["Loss", "Top1_Accuracy", "Top5_Accuracy"],
            global_step_transform=global_step_from_engine(trainer)
        )

    def score_function(engine):
        return engine.state.metrics["Top1_Accuracy"]

    objects_to_checkpoint = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
    }

    checkpoint = Checkpoint(
        objects_to_checkpoint,
        ClearMLSaver(dirname=config.dir_output+'/checkpoint/checkpoint_DS', require_empty=False),
        n_saved=1, score_function=score_function,
        score_name="test_acc", filename_prefix="best",
        global_step_transform=global_step_from_engine(trainer)
    )

    trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
    test_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)
    trainer.run(train_loader, config.max_epochs)
    tb_logger.close()


if __name__ == "__main__":

    from clearml import Task

    parser = argparse.ArgumentParser("STL 10 SimCLR")
    parser.add_argument("--backend", default="gloo",
                        type=str, help="Backend for distributed computation")
    parser.add_argument("--dataset",
                        default="DRPDataset2D",
                        type=str, help="Type dataset for training simclr")
    parser.add_argument("--version_dataset", default='1.1', type=str, help="Version drpdataset")
    parser.add_argument("--root_data", default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/scripts/data",
                        type=str, help="Path to dataset")
    # parser.add_argument("--root_dataset_train", default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/scripts/data",
    #                     type=str, help="Path to dataset")
    # parser.add_argument("--root_dataset_test",
    #                     default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/scripts/data",
    #                     type=str, help="Path to dataset")

    parser.add_argument("--dir_fturn",
                        default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/checkpoint/checkpoint_DS/best_checkpoint_10_test_acc=0.4426.pt",
                        type=str, help="Path to trained model")
    parser.add_argument("--optimizer_name", default='Adam', type=str, help="Optimizer")
    parser.add_argument("--num_worker", default=2, type=int,
                        help="Number of workers over the dataset to train")
    parser.add_argument("--max_epochs", default=10, type=int,
                        help="Number of sweeps over the dataset to train")
    parser.add_argument("--batch_size", default=10, type=int,
                        help="Number of images in each mini-batch")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
    parser.add_argument("--model_name", default="resnet50", type=str, help="Base model")
    parser.add_argument("--nb_classes", default=9, type=int,
                        help="Number class in dataset")
    parser.add_argument("--dir_output", default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage",
                        type=str, help="directory of output which includes checkpoint and tensorboard")
    args = parser.parse_args()

    task = Task.init(
        project_name="DRP",
        task_name=" SimCLR on " + args.__getattribute__("dataset")
    )
    with idist.Parallel(backend=args.backend) as parallel:
        parallel.run(training, args)
