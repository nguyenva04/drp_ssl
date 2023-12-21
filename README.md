# Representation learning with SimCLR 

## Installation
Install from IFPEN GitLab repository:

    pip install git+https://gitlab.ifpen.fr/DigitalSandbox/ai_ignite/simclr.ignite_stage.git

Install from sources:

    pip setup.py install

Install the dependencies from `pip`:

    pip install -r requirements.text
    
## Configuration summary
* Describe some configuration parameters needed to train SimCLR. 
- `backend`(default `gloo`): Backend for distributed computation - `nccl` for the GPU and `gloo` for the CPU.
- `dataset`(default `DRPDataset2D`): Dataset for training SimCLR(Should be `stl10` or `DRPDataset2D`).
- `data_root`: Path to dataset on the datalake `islin-hdplnod05`.
- `max_epochs`(default 10): Number of images in each mini-batch.
- `model`(default ``resnet50): Base Encoder ConvNet for feature extraction.
- `lr`(default `0.0003`).
- `dir_output`: directory to save trained model and tensorboard.
* Describe some configuration parameters needed to train Downstream task.
(we only mention the special params for Downstream task)
- `dir_pretrained`: Path to pretrained backbone 
- `model_name`(default `resnet50`): model name must be the same when training SimCLR
- `mode_evaluate`(default `Linear Evaluation`): Should be `Linear Evaluation` or `Fine Tune`
- `nb_images`(default 1000): Number labeled data per class for train downstrem
## Training SimCLR
Example of a sequential train:

    python scripts/training.py --max_epochs=5 --dataset=stl10 --model=resnet50
    
    python scripts/downstream_training.py --max_epochs=5 --model_name=resnet50 --mode_evaluate=LinearEvaluation
    
## Downstream task
After training SimCLR to obtain good representation vectors, we wil take the pretrained backbone and only add one linear layer on top. To evaluate the quality of representation vectors, we fine tune the whole network ou only train the last linear layer on fixed backbone.


