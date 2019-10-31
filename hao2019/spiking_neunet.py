from typing import Dict, List

import torch
import numpy as np

from tqdm import tqdm

from bindsnet.encoding import PoissonEncoder
from bindsnet.network.network import Network
from bindsnet.network.monitors import Monitor

from utils import load_data, sl_poisson, prediction

"""

    spikes: Dict[str, torch.Tensor],
    spike_record: torch.Tensor,
    :param spikes: Monitor for spikes.
    :param spike_record: Tensor of shape ``[update_interval, time, n_outpt]``
    for recording spikes during the simulation."""

class Spiking:
    """
        
    """

    def __init__(
        self,
        network: Network,
        n_outpt: int,
        dataset_name: str = "MNIST",
        seed: int = 0,
        n_workers: int = -1,
        time: int = 350,
        dt: float = 0.5,
        intensity: float = 128,
        update_interval: int = 250,
        gpu: bool = False,
    ) -> None:
        """
        Testing network with test data.

        :param network: Network model to use.
        :param dataset: Name of dataset to use.
        :param n_workers: Number of workers to use.
        :param time: Length of Poisson spike train per input variable.
        :param dt: Simulation time step.
        :param update_interval: Interval to compute network accuracy.
        :param gpu: Whether to use gpu.
        """
        super().__init__()

        self.network = network
        self.n_workers = n_workers
        self.time = time
        self.dt = dt
        self.update_interval = update_interval
        self.gpu = gpu

        self.test_acc_history = []

        # Determines number of workers to use.
        if n_workers == -1:
            self.n_workers = gpu * 4 * torch.cuda.device_count()

        # Sets up Gpu use.
        if gpu:
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        # Directs network to GPU.
        if gpu:
            network.to("cuda")

        # Load train & test data.
        encoder = PoissonEncoder(time=time, dt=dt)
        self.train_dataset = load_data(dataset_name, encoder, True, intensity)
        self.test_dataset = load_data(dataset_name, encoder, False, intensity)

        # Voltage recording for excitatory layer.
        exc_voltage_monitor = Monitor(network.layers["Y"], ["v"], time=time)
        network.add_monitor(exc_voltage_monitor, name="exc_voltage")

        # Set up monitors for spikes and voltages.
        spikes = {}
        for layer in set(network.layers):
            spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
            network.add_monitor(spikes[layer], name="%s_spikes" % layer)

        voltages = {}
        for layer in set(network.layers) - {"X"}:
            voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
            network.add_monitor(voltages[layer], name="%s_voltages" % layer)

        self.spikes = spikes
        self.voltages = voltages

        # Record spikes during the simulation.
        self.spike_record = torch.zeros(update_interval, time, n_outpt)

    def train_network(self, shuffle: bool = True) -> None:
        """
        Train the spiking neural network.

        :param shuffle: Whether to shuffle the dataset.
        """
        pass

        network = self.network

        # Change training mode of network to False.
        network.train(True)

        # Create a dataloader for train data.
        dataloader = get_dataloader(self.train_dataset, shuffle=shuffle)

    def test_network(
        self,
        n_samples: int = None,
        mode: str = "test",
        shuffle: bool = True,
    ) -> None:
        """
        Test the spiking neural network.

        :param n_samples: Number of samples to use from dataset for testing.
        :param mode: Determine (train / validation / test) dataset to use for testing.
        :param shuffle: Whether to shuffle the dataset.
        """

        if n_samples is not None:
            assert (n_samples > 0), "Number of samples must be greater than 0"

        mode_list = ['train', 'validation', 'test']
        if mode.lower() not in mode_list:
            msg = "\"%s\" is not defined. Available mode: %s" % (mode, mode_list)
            raise NameError(msg)
        else:
            mode = mode.capitalize()

        network = self.network
        timestep = int(self.time/self.dt)

        # Change training mode of network to False.
        network.train(False)

        # TODO
        # Train / Validation / Test dataset based on mode?

        # Create a dataloader for test data.
        dataloader = self.get_dataloader(self.test_dataset, shuffle=shuffle)

        labels = []
        accuracy = []

        for step, batch in enumerate(tqdm(dataloader)):
            # Calculate network accuracy at every update interval.
            if step % self.update_interval == 0 and step > 0:
                self.cal_acc(mode, labels, accuracy)

                labels = []

            # Get next input sample.
            inputs = {"X": batch["encoded_image"].view(timestep, 1, 1, 28, 28)}

            if self.gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Run the network on the input.
            network.run(inputs=inputs, time=self.time, input_time_dim=1)

            # Save label of each sample.
            labels.append(batch["label"])

            # Add to spikes recording.
            spikes = self.spikes["Z"].get("s").squeeze()
            self.spike_record[step % self.update_interval] = spikes

            network.reset_state_variables()  # Reset state variables.

            if n_samples is not None and step == n_samples:
                break

        self.cal_acc(mode, labels, accuracy)
        if mode == "Test":
            self.test_acc_history.append(np.mean(accuracy))

    def get_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        """
        Generate dataloader for the dataset.

        :param dataset: Dataset to use.
        :param shuffle: Whether to shuffle the dataset.
        :return: Return dataloader.
        """

        # Create a dataloader to iterate and batch data.
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            num_workers=self.n_workers,
            pin_memory=self.gpu,
        )

        return dataloader

    def cal_acc(
        self,
        mode: str,
        labels: List[torch.Tensor],
        accuracy: List[float],
    ) -> None:
        """
        Calculate network accuracy by comparing the labels from input samples and 
        labels from SL neurons.

        :param mode: Name of the mode.
        :param labels: Labels of the input samples.
        :param accuracy: Used to record network accuracy for each update interval.
        """

        # Convert the array of labels into a tensor
        label_tensor = torch.tensor(labels)

        # Get network predictions.
        test_pred = prediction(spikes=self.spike_record)

        # Adjust length of predictions array to the same as labels array.
        if len(test_pred) > len(label_tensor):
            test_pred = test_pred[:len(label_tensor)]

        # Compute network accuracy.
        accuracy.append(
            100
            * torch.sum(label_tensor.long() == test_pred).item()
            / len(label_tensor)
        )

        print(
            "\n%s accuracy: %.2f (last), %.2f (average), %.2f (best)"
            % (
                mode,
                accuracy[-1],
                np.mean(accuracy),
                np.max(accuracy),
            )
        )
