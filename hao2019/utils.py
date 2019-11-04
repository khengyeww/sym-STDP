from typing import List

import os
import torch

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


def transform_image(
    image: torch.Tensor, intensity_scale: int, old_intensity: float
) -> torch.Tensor:
    """
    Transform image pixel intensity (which equals to firing rates).

    :param image: Tensor of shape ``[batch_size, *input_shape]``
        of the original image's pixel intensity.
    :param intensity_scale: Scale for increasing the pixel intensity.
    :param old_intensity: Maximum pixel intensity of the original image.
    :return: Tensor of shape ``[batch_size, *input_shape]`` of transformed image.
    """
    intensity = 256.0 / 8.0 * intensity_scale
    transform = transforms.Compose(
        [transforms.Lambda(lambda x: x * intensity / old_intensity)]
    )
    new_image = transform(image)

    return new_image


def get_network_const(n_neurons: int, default_value: List[float]) -> float:
    """
    Set time constant of threshold potential decay & decay factor
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
