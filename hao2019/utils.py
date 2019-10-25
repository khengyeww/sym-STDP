import os
import torch

from torchvision import transforms

from bindsnet.datasets import *


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
    try:

        if dataset_name == "MNIST":
            # Load MNIST data.
            dataset = MNIST(
                encoder,
                None,
                root=os.path.join("..", "data", dataset_name),
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
                root=os.path.join("..", "data", dataset_name),
                train=train,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
                ),
            )

        return dataset

    except:
        raise NameError("Name %s is not defined" %data_name)
        #raise NameError("name {} is not defined".format(data_name))


def get_network_const(n_neurons: int, default_value: []) -> float:
    """
    Set time constant of threshold potential decay & decay factor
    for different sized network.

    :param n_neurons: Number of excitatory, inhibitory neurons.
    :param default_value: Array of default value for
        constant tc_theta_decay, theta_plus.
    :return: Return constant tc_theta_decay, theta_plus.
    """
    # Num. of neurons : (theta time constant, alpha decay factor)
    const_choices = {
        100  : (6e6, 8.4e5),
        400  : (6e6, 8.4e5),
        1600 : (8e6, 1.12e6),
        6400 : (2e7, 2e6),
        10000: (2e7, 2e6),
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
    rate[datum != 0] = 200

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

    return spikes.view(time, *shape)


def prediction(spikes: torch.Tensor) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with highest spiking activity over all neurons
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
