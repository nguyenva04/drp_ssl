import argparse

import ignite.distributed as idist
import torch
import torch.nn as nn
from clearml import Dataset
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.clearml_logger import *
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Events, create_supervised_evaluator, Engine
from ignite.handlers import Checkpoint
from ignite.metrics import Accuracy, Loss, RunningAverage, TopKCategoricalAccuracy
from torch.optim import SGD, Adam
from torchvision import transforms
from torchvision.models.resnet import resnet18, resnet50, resnet101, resnet152
from drp.twod.drp2d import DRP2D


def training(_, config):
    assert config.model in ['resnet18', 'resnet50', 'resnet101', 'resnet152'], (
        f'model "{config.model}" is not supported.'
        'Should be "resnet18" or "resnet50".')

    resnet = None
    if config.model == 'resnet18':
        resnet = resnet18(pretrained=False)
    if config.model == 'resnet50':
        resnet = resnet50(pretrained=False)
    if config.model == 'resnet101':
        resnet = resnet101(pretrained=False)
    if config.model == 'resnet152':
        resnet = resnet152(pretrained=False)
    if config.pretrain == "True":
        # Load pretrained ResNet from ClearML dataset
        model_path = Dataset.get(
            dataset_name=config.model,
            dataset_project="DRP").get_local_copy()

        file_name = f"{config.model}.pth"
        state_dict = torch.hub.load_state_dict_from_url(
            'http://dummy', model_dir=model_path, file_name=file_name)
        resnet.load_state_dict(state_dict)

    nb_features = resnet.fc.in_features
    resnet.fc = nn.Linear(nb_features, config.nb_classes)
    resnet = idist.auto_model(resnet)

    optimizer_dict = {
        'Adam': Adam(params=resnet.parameters(),
                     lr=config.lr,
                     weight_decay=1e-4),
        'SGD': SGD(params=resnet.parameters(),
                   lr=config.lr,
                   momentum=0.9)
    }
    criterion = nn.CrossEntropyLoss(reduction='mean')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_dict[config.optimizer_name], step_size=20, gamma=0.1)
    lr_scheduler = LRScheduler(scheduler)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])
    dataset_train = DRP2D(root=config.root_data, download=True, train=False, version="1.0", host='islin-hdpmas1',
                          port=5431, transform=transform)
    dataset_test = DRP2D(root=config.root_data, download=True, train=False, version="1.0", host='islin-hdpmas1',
                         port=5431, transform=transform)

    train_loader = idist.auto_dataloader(dataset_train,
                                         batch_size=config.batch_size, num_workers=config.num_worker,
                                         shuffle=True, drop_last=True)
    test_loader = idist.auto_dataloader(dataset_test,
                                        batch_size=config.batch_size, num_workers=config.num_worker,
                                        shuffle=False, drop_last=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    metrics = {
        "Top1_Accuracy": Accuracy(),
        "Loss": Loss(criterion),
        "Top3_Accuracy": TopKCategoricalAccuracy(k=3)
    }
    optimizer = optimizer_dict[config.optimizer_name]

    def train_step(_, batch):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        resnet.train()
        y_pred = resnet(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    # Define a trainer engine
    trainer = Engine(train_step)

    train_evaluator = create_supervised_evaluator(resnet, metrics=metrics, device=device)
    test_evaluator = create_supervised_evaluator(resnet, metrics=metrics, device=device)
    tb_logger = TensorboardLogger(log_dir=config.dir_output + "/tensorboard/tensorboad_DS")
    # tb_logger = ClearMLLogger(
    #     project_name="DRP",
    #     task_name="Supervised training + Pretrained:" + config.pretrain
    # )

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
            f"Top3-Acc: {metrics['Top3_Accuracy']:.3f} Avg loss: {metrics['Loss']:.2f}")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results():
        test_evaluator.run(test_loader)
        metrics = test_evaluator.state.metrics
        print(
            f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['Top1_Accuracy']:.3f} "
            f"Top3-Acc: {metrics['Top3_Accuracy']:.3f} Avg loss: {metrics['Loss']:.2f}")

    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer_dict[config.optimizer_name], param_name="lr"),
        event_name=Events.ITERATION_STARTED,
    )

    for tag, evaluator in [("training", train_evaluator), ("testing", test_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator, event_name=Events.EPOCH_COMPLETED,
            tag=tag, metric_names=["Loss", "Top1_Accuracy", "Top3_Accuracy"],
            global_step_transform=global_step_from_engine(trainer)
        )

    def score_function(engine):
        return engine.state.metrics["Top1_Accuracy"]

    objects_to_checkpoint = {"model": resnet}

    checkpoint = Checkpoint(
        objects_to_checkpoint,
        ClearMLSaver(dirname='checkpoint', require_empty=False),
        n_saved=1, score_function=score_function,
        score_name="Testing_accuracy", filename_prefix="best",
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
    parser.add_argument("--root_data", default="C:/Users/nguyenva/Documents/SimCLR_stage/simclr.ignite_stage/scripts/data",
                        type=str, help="Path to STL10 dataset")
    parser.add_argument("--pretrain", default="True",
                        type=str, help="load pretrained model in ImageNet or not")
    parser.add_argument("--optimizer_name", default='Adam', type=str, help="Optimizer")
    parser.add_argument("--model", default="resnet50", type=str, help="Base model")
    parser.add_argument("--num_worker", default=2, type=int,
                        help="Number of workers over the dataset to train")
    parser.add_argument("--max_epochs", default=10, type=int,
                        help="Number of sweeps over the dataset to train")
    parser.add_argument("--batch_size", default=10, type=int,
                        help="Number of images in each mini-batch")
    parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
    parser.add_argument("--nb_classes", default=9, type=int,
                        help="Number class in dataset")
    parser.add_argument("--dir_output", default="C:/Users/nguyenva/Documents/simclr.ignite_stage/simclr_stl", type=str,
                        help="directory of output which includes checkpoint and tensorboard")
    args = parser.parse_args()

    task = Task.init(
        project_name="DRP",
        task_name="ResNet50 supervised learning"
    )

    with idist.Parallel(backend=args.backend) as parallel:
        parallel.run(training, args)
