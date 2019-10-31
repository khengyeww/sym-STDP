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
        print('+-' + '-' * width + '-+')
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


def make_dirs(paths: List[str]) -> None:
    """
    Setup directories within path.

    :param paths: List of paths.
    """
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
        # Alternative way:
        # os.makedirs(path, exist_ok=True)


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


def sl_poisson(datum: torch.Tensor, time: int, dt: float = 1.0) -> torch.Tensor:
    # language=rst
    """
    Generates Poisson-distributed spike trains for SL neurons according to
    one-hot encoding scheme. Inputs must be non-negative, and give the
    firing rate in Hz. Inter-spike intervals (ISIs) for non-negative data
    incremented by one to avoid zero intervals while maintaining ISI distributions.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"

    # Get shape and size of data.
    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()
    time = int(time / dt)

    # Set 200Hz of firing rate for one SL neuron based on label.
    rate = torch.zeros(size)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time + 1]))
    intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

    # Calculate spike times by cumulatively summing over time dimension.
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time + 1] = 0

    # Create tensor of spikes.
    spikes = torch.zeros(time + 1, size).byte()
    spikes[times, torch.arange(size)] = 1
    spikes = spikes[1:]

    # print("spikes in utils!!!!!!!!!!!!!!!!!")

    return spikes.view(time, *shape)


def prediction(spikes: torch.Tensor) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with highest spiking activity over SL neurons
    during testing mode.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of 
        a layer's spiking activity.
    :return: Predictions tensor of shape ``(n_samples)`` resulting from
        the classification scheme.
    """
    n_samples = spikes.size(0)

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    # Predictions are arg-max of layer-wise firing rates.
    predictions = torch.sort(spikes, dim=1, descending=True)[1][:, 0]

    return predictions
