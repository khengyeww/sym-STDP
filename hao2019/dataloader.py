import os
import torch

from torchvision import transforms

from bindsnet.datasets import *


def load_data(dataset_name: str, encoder: torch.Tensor, intensity: float) -> torch.utils.data.Dataset:

    try:

        if dataset_name == "MNIST":
            # Load MNIST data.
            dataset = MNIST(
                encoder,
                None,
                root=os.path.join("..", "data", dataset_name),
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
                ),
            )
        elif dataset_name == "FashionMNIST":
            # Load FashionMNIST data.
            dataset = FashionMNIST(
                encoder,
                None,
                root=os.path.join("..", "data", dataset_name),
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
                ),
            )

        return dataset
    
    except:
        raise NameError("Name %s is not defined" %data_name)
        #raise NameError("name {} is not defined".format(data_name))