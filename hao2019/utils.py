from typing import Union, Tuple, List, Sequence

import os
import torch
import numpy as np

from torchvision import transforms
from textwrap import wrap

from bindsnet.datasets import *


def msg_wrapper(msg: List[str], style: int) -> None:
    """
    Wrap the message with a border.

    :param msg: List of messages.
    :param style: Pick the style of the border.
    """
    width = max([len(sentence) for sentence in msg])

    if style == 1:
        print('\n' + '#' * width)
        for sentence in msg:
            for line in wrap(sentence, width):
                print('{0:^{1}}'.format(line, width))
        print('#' * width + '\n')
    elif style == 2:
        print('\n+-' + '-' * width + '-+')
        for sentence in msg:
            for line in wrap(sentence, width):
                print('| {0:^{1}} |'.format(line, width))
        print('+-' + '-'*(width) + '-+' + '\n')


def load_data(
    dataset_name: str,
    encoder: torch.Tensor = None,
    train: bool = True,
    intensity: float = 128,
) -> torch.utils.data.Dataset:
    """
    Load dataset of choice.

    :param dataset_name: Name of dataset.
    :param encoder: Spike encoder for generating spike trains.
    :param train: True for train data, False for test data.
    :param intensity: Intensity for transformation of data.
    :return: Return dataset.
    """
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(ROOT_PATH, "..", "data", dataset_name)

    try:

        if dataset_name == "MNIST":
            # Load MNIST data.
            dataset = MNIST(
                encoder,
                None,
                root=DATA_PATH,
                train=train,
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
                root=DATA_PATH,
                train=train,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
                ),
            )

        return dataset

    except:
        raise NameError("Name \"%s\" is not defined" % dataset_name)
        #raise NameError("name \"{}\" is not defined".format(dataset_name))


def make_dirs(path: str) -> None:
    """
    Setup directories within path.

    :param path: Name of path.
    """
    os.makedirs(path, exist_ok=True)
    # Alternative way:
    # if not os.path.isdir(path):
    #     os.makedirs(path)


def get_network_const(n_neurons: int, default_value: List[float]) -> float:
    """
    Get time constant of threshold potential decay & decay factor
    for different sized network.

    :param n_neurons: Number of excitatory, inhibitory neurons.
    :param default_value: Array of default value for
        constant theta_plus, tc_theta_decay.
    :return: Return constant theta_plus, tc_theta_decay.
    """
    # Num. of neurons : (theta plus, theta decay time constant)
    const_choices = {
        100  : (0.07, 6e6),
        400  : (0.07, 6e6),
        1600 : (0.07, 8e6),
        6400 : (0.05, 2e7),
        10000: (0.05, 2e7),
    }
    const = const_choices.get(n_neurons, default_value)

    return const[0], const[1]


def get_lrate(
    n_neurons: int,
    default_lrates: Tuple[Union[float, Sequence[float]]],
) -> float:
    """
    Get learning rate for different sized network.

    :param n_neurons: Number of excitatory, inhibitory neurons.
    :param default_lrates: Default value of learning rates.
    :return: Return pre- and post-synaptic learning rates for each layer.
    """
    # Num. of neurons : (nu_exc, nu_sl)
    lrate_choices = {
        100  : (5e-3, 4e-2),
        400  : (5e-3, 4e-2),
        1600 : (2e-2, 8e-2),
        6400 : (1e-2, 4e-2),
        10000: (7.5e-3, 4e-2),
    }
    lrate = lrate_choices.get(n_neurons, default_lrates)

    # lrate[0]: learning rates for exc layer, lrate[1]: learning rates for SL layer.
    return lrate[0], lrate[1]


def sample_from_class(
    dataset: torch.utils.data.Dataset,
    n_samples: int,
) -> torch.utils.data.Dataset:
    """
    Stratified sampling. Create a dataset with ``n_samples`` numbers of each class
    from the original dataset.

    :param dataset: Original dataset.
    :param n_samples: Number of samples to use from each class.
    """
    class_counts = {}
    new_data = []
    new_labels = []

    for index, data in enumerate(dataset.data):
        label = dataset.targets[index]
        c = label.item()
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= n_samples:
            new_data.append(torch.unsqueeze(data, 0))
            new_labels.append(torch.unsqueeze(label, 0))

    dataset.data = torch.cat(new_data)
    dataset.targets = torch.cat(new_labels)

    return dataset
