import io
import torch
import requests
import numpy as np
import torchvision
import PIL.Image as Image
from torchvision.datasets import STL10
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


def load_data(data_root, dataset_name, transform, split="train", train=True):
    """
    Load dataset behind a proxy
    """
    from six.moves import urllib

    proxy = urllib.request.ProxyHandler({'http': 'irproxy:8082', "https": 'irproxy:8082'})
    # construct a new opener using your proxy settings
    opener = urllib.request.build_opener(proxy)

    # install the openen on the module-level
    urllib.request.install_opener(opener)
    if dataset_name == "stl10":
        dataset = STL10(root=data_root, download=True, split=split, transform=transform)
    elif dataset_name == "cifar10":
        dataset = CIFAR10(root=data_root, train=train, transform=transform, download=True)
    return dataset


class DRPDataset2D(Dataset):
    def __init__(self, path, root_url, split='train+unlabeled', transform=None):
        super(DRPDataset2D, self).__init__()
        self.type_rock = [4283, 4287, 4288, 4290, 4296, 4302, 4308, 4318, 4322]
        self.path = path
        self.data = []
        self.root_url = root_url
        self.transform = transform
        files = self._get_file_list(self.path)

        for img_path in files:
            label = self._get_label(img_path)
            url = self.root_url + "/file"
            header = {
                'filename': img_path,
                'Content-type': 'image/jpeg'
            }
            response = requests.get(url, headers=header)
            code = response.status_code

            if code == 200:
                img = Image.open(io.BytesIO(response.content))
                img = np.array(img).astype(np.float32)/255.0
                if self.transform is not None:

                    img = self.transform(img)
                    img = [i.float() for i in img]
                    label = torch.tensor(label)
                    label = label.to(dtype=torch.float)
                else:
                    tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
                    img = tensor(img).float()
                    label = torch.tensor(label)
                    label = label.to(dtype=torch.float)

            self.data.append([img, label])

    def _get_label(self, img_path):
        img_path = img_path.replace(self.path, '')
        number_in_path = list(map(int, ''.join([x if x.isdigit() else ' ' for x in img_path]).split()))
        return self.type_rock.index(number_in_path[0])

    def _get_file_list(self, img_path):
        url = self.root_url + "/list"
        header = {
            'img_path': img_path
        }
        response = requests.get(url, headers=header)
        code = response.status_code
        print("Status Code", code)
        if code == 200:
            files = response.json()
            return files
        else:
            raise Exception(f"unable to find any file at {url}({img_path})")

    def __getitem__(self, item):

        [img, label] = self.data[item]

        return img, label

    def __len__(self):

        return len(self.data)


def get_smaller_dataset(dataset, nb_images):
    indices = np.random.randint(low=0, high=len(dataset), size=nb_images)
    subset = torch.utils.data.Subset(dataset, indices)
    return subset


if __name__ == "__main__":
    from simclr_stl.simclr.transformation.PairTransform import PairTransform
    from torchvision.transforms import transforms

    transform = transforms.ToTensor()
    dataset = DRPDataset2D(path='/R11-data/drp-data/db_2D/128_data/train1/', root_url='http://localhost:5234',
                           transform=None)
    print(len(dataset))
