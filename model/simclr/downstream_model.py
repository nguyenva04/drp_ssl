import torch
import torch.nn as nn
import torch.nn.functional as f


class DSModel(torch.nn.Module):
    def __init__(self, pre_model, nb_classes, dir_pretrained=None):
        super().__init__()
        if dir_pretrained is not None:
            checkpoint = torch.load(dir_pretrained)
            self.pre_model = pre_model
            self.pre_model.load_state_dict(checkpoint["model"], strict=False)
            print('load pretraining success')
        else:
            self.pre_model = pre_model
        self.nb_features = self.pre_model.nb_features
        self.nb_classes = nb_classes
        self.features = nn.Sequential(*list(self.pre_model.children())[:-1])
        self.fc1 = nn.Linear(self.nb_features, self.nb_features, bias=False)
        self.fc2 = nn.Linear(self.nb_features, 256, bias=False)
        self.fc3 = nn.Linear(256, self.nb_classes, bias=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)


    def forward(self, x):
        out, _ = self.pre_model(x)
        out = self.fc1(self.dropout(out))
        out = self.fc2(self.dropout(out))
        out = self.fc3(out)
        # out = self.fc2(self.dropout(self.relu(self.fc1(self.dropout(out)))))
        return out


if __name__ == "__main__":
    from Encoder import BaseEncoder

    base_encoder = BaseEncoder(model='resnet18', feature_dim=128)
    base_encoder = DSModel(base_encoder, 10,
                           dir_pretrained="C:/Users/nguyenva/Documents/simclr.ignite_stage/simclr_stl/checkpoint/best_checkpoint_2_Testing_accuracy=0.6960.pt")
    # base_encoder.projector = nn.Linear(512, 10)
    print(base_encoder)
