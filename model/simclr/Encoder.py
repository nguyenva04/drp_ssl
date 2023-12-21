import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18, resnet101, resnet152
from torchvision.models import vgg16
from clearml import Task, Dataset


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ModifiedResNet(nn.Module):
    def __init__(self, model_name, feature_dim=128):
        super(ModifiedResNet, self).__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim
        self.resnet_dict = {"resnet18": resnet18(pretrained=False),
                            "resnet50": resnet50(pretrained=False),
                            "resnet101": resnet101(pretrained=False),
                            "resnet152": resnet152(pretrained=False),
                            "vgg16": vgg16(pretrained=False)
                            }

        resnet = self.resnet_dict[model_name]

        model_path = Dataset.get(
            dataset_name=self.model_name,
            dataset_project="DRP").get_local_copy()

        file_name = f"{self.model_name}.pth"
        state_dict = torch.hub.load_state_dict_from_url('http://dummy', model_dir=model_path, file_name=file_name)
        resnet.load_state_dict(state_dict)
        if model_name == "vgg16":
            weights = resnet.state_dict()['features.0.weight'].mean(dim=1, keepdim=True)
            self.n_features = resnet.classifier[0].in_features
            resnet.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            resnet.state_dict()['features.0.weight'] = weights
            self.features = nn.Sequential(*list(resnet.children()))
            self.features[2] = Identity()
            self.projector = nn.Sequential(
                nn.Linear(self.n_features, 4096, bias=False),
                nn.BatchNorm1d(4096),
                nn.ReLU(),
                nn.Linear(4096, self.feature_dim, bias=False),
                nn.BatchNorm1d(self.feature_dim),
            )
        else:
            weights = resnet.state_dict()['conv1.weight'].mean(dim=1, keepdim=True)
            self.n_features = resnet.fc.in_features
            resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            resnet.state_dict()['conv1.weight'] = weights
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.projector = nn.Sequential(
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.BatchNorm1d(self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, self.feature_dim, bias=False),
                nn.BatchNorm1d(self.feature_dim),
                )

    def forward(self, x):
        x = self.features(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.projector(feature)
        return nn.functional.normalize(feature, dim=-1), nn.functional.normalize(out, dim=-1), feature

