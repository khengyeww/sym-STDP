from typing import Dict, List

import os
import torch
import numpy as np

from tqdm import tqdm

from bindsnet.encoding import poisson
from bindsnet.encoding import PoissonEncoder
from bindsnet.network.network import Network
from bindsnet.network.monitors import Monitor

from plot import Plot
from utils import load_data, make_dirs, transform_image, msg_wrapper


class Spiking:
    """
    Class to maintain training and testing functions for spiking neural network (SNN).
    """

    def __init__(
        self,
        network: Network,
        n_outpt: int,
        results_path: str,
        dataset_name: str = "MNIST",
        seed: int = 0,
        n_workers: int = -1,
        time: int = 350,
        dt: float = 0.5,
        update_interval: int = 250,
        plot: bool = False,
        gpu: bool = False,
    ) -> None:
        """
        Testing network with test data.

        :param network:         Network model to use.
        :param n_outpt:         Number of output (SL) neurons.
        :param results_path:    Path to save training & testing results.
        :param dataset_name:    Name of dataset to use.
        :param seed:            Seed for pseudorandom number generator (PRNG).
        :param n_workers:       Number of workers to use.
        :param time:            Length of Poisson spike train per input variable.
        :param dt:              Simulation time step.
        :param update_interval: Interval to show network accuracy.
        :param plot:            Whether to plot for visualization of network training.
        :param gpu:             Whether to use gpu.
        """
        super().__init__()

        self.network = network
        self.n_outpt = n_outpt
        self.results_path = results_path
        self.n_workers = n_workers
        self.time = time
        self.dt = dt
        self.update_interval = update_interval
        self.plot = plot
        self.gpu = gpu

        timestep = int(time / dt)
        self.timestep = timestep

        self.start_intensity_scale = 2
        intensity = 256.0 / 8.0 * self.start_intensity_scale
        self.start_intensity = intensity

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.train_spike = []
        self.test_spike = []

        self.right_pred = []
        self.wrong_pred = []
        self.acc_history = {'train_acc': [], 'test_acc': []}

        self.plotf = Plot()
        self.exc_init_weight = network.connections[("X", "Y")].w.detach().clone()
        self.sl_init_weight = network.connections[("Y", "Z")].w.detach().clone()
        self.exc_final_weight = None
        self.sl_final_weight = None

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

        # Setup directories within path.
        make_dirs(results_path)

        # Set up monitors for spikes and voltages.
        spikes = {}
        for layer in set(network.layers):
            l = network.layers[layer]
            spikes[layer] = Monitor(l, state_vars=["s"], time=timestep)
            network.add_monitor(spikes[layer], name="%s_spikes" % layer)

        voltages = {}
        for layer in set(network.layers) - {"X"}:
            l = network.layers[layer]
            voltages[layer] = Monitor(l, state_vars=["v"], time=timestep)
            network.add_monitor(voltages[layer], name="%s_voltages" % layer)

        self.spikes = spikes
        self.voltages = voltages

    def train_network(self, n_samples: int = None, shuffle: bool = False) -> None:
        """
        Train the spiking neural network.

        :param n_samples: Number of samples to use from dataset for training.
        :param shuffle: Whether to shuffle the dataset.
        """
        if n_samples is not None:
            assert (n_samples > 0), "Samples for training must be greater than 0"

            dataset = torch.utils.data.random_split(
                self.train_dataset, [n_samples, len(self.train_dataset) - n_samples]
            )[0]
        else:
            dataset = self.train_dataset

        # Create a dataloader to iterate and batch data.
        dataloader = self.get_dataloader(dataset, shuffle=shuffle)

        network = self.network

        # Change training mode of network to True.
        network.train(True)

        progress = tqdm(dataloader)
        for step, batch in enumerate(progress):
            # if batch["label"] != 9:
            #     # print(batch["label"])
            #     continue

            # Generate 0Hz or 200Hz Poisson rates for SL neurons in training mode.
            sl_label = torch.zeros(self.n_outpt)
            sl_label[batch["label"]] = 200
            sl_spike = poisson(datum=sl_label, time=self.time, dt=self.dt)

            # Get next input sample & SL neurons one-hot spikes.
            inputs = {"X": batch["encoded_image"].view(self.timestep, 1, 1, 28, 28)}
            clamp = {"Z": sl_spike}
            if self.gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}
                clamp = {k: v.cuda() for k, v in clamp.items()}

            # Run the network on the input.
            network.run(inputs=inputs, time=self.time, input_time_dim=1, clamp=clamp)

            # Calculate number of spikes from excitatory neurons.            
            exc_spike = self.spikes["Y"].get("s").squeeze()
            exc_spike_count = exc_spike.sum()
            # Alternative way:
            # exc_spike_count = torch.sum(torch.sum(exc_spike, dim=0), dim=0)

            if exc_spike_count < 5:
                intensity_scale = self.start_intensity_scale

            # Re-present the input sample with increased firing rate
            # if excitatory neurons fire less than five spikes.
            while exc_spike_count < 5 and intensity_scale < 32:
                intensity_scale += 1
                network.reset_state_variables()

                # Get new generated spikes.
                new_image = transform_image(
                    batch["image"], intensity_scale, self.start_intensity
                )
                new_encoded_image = poisson(datum=new_image, time=self.time, dt=self.dt)

                inputs = {"X": new_encoded_image.view(self.timestep, 1, 1, 28, 28)}
                if self.gpu:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                network.run(inputs=inputs, time=self.time, input_time_dim=1, clamp=clamp)
                exc_spike_count = self.spikes["Y"].get("s").squeeze().sum()

            if self.plot:
                pass
                # plot_every_step()

            self.train_spike.append(batch["label"])
            self.train_spike.append(self.spikes["Z"].get("s").squeeze().sum(0))

            network.reset_state_variables()  # Reset state variables.

        self.exc_final_weight = network.connections[("X", "Y")].w.detach().clone()
        self.sl_final_weight = network.connections[("Y", "Z")].w.detach().clone()

    def test_network(
        self, n_samples: int = None, data_mode: str = "test", shuffle: bool = True
    ) -> None:
        """
        Test the spiking neural network.

        :param n_samples: Number of samples to use from dataset for testing.
        :param data_mode: Specifies (train / validation / test) dataset
            to use for testing.
        :param shuffle: Whether to shuffle the dataset.
        """
        if n_samples is not None:
            assert (n_samples > 0), "Samples for testing must be greater than 0"

            dataset = torch.utils.data.random_split(
                self.test_dataset, [n_samples, len(self.test_dataset) - n_samples]
            )[0]
        else:
            dataset = self.test_dataset

        # Check for the mode selected.
        mode_list = ['train', 'validation', 'test']
        if data_mode.lower() not in mode_list:
            print("\"%s\" is not defined. Available mode: %s" % (data_mode, mode_list))
            print("Default to using test dataset.\n")
            data_mode = "Test"
        else:
            print("!!! Testing the network with %s data. !!!\n" % data_mode.lower())
            data_mode = data_mode.capitalize()

        # TODO
        # Train / Validation / Test dataset based on mode?

        # Create a dataloader for test data.
        dataloader = self.get_dataloader(dataset, shuffle=shuffle)

        network = self.network

        # Change training mode of network to False.
        network.train(False)

        correct_pred = 0
        accuracy = []

        progress = tqdm(dataloader)
        for step, batch in enumerate(progress):
            # if batch["label"] != 9:
            #     # print(batch["label"])
            #     continue

            # Calculate network accuracy at every update interval.
            if step % self.update_interval == 0 and step > 0:
                tmp_acc = 100 * correct_pred / step
                accuracy.append(tmp_acc)
                print(
                    "\n%s accuracy: %.2f (last), %.2f (average), %.2f (best)"
                    % (
                        data_mode,
                        accuracy[-1],
                        np.mean(accuracy),
                        np.max(accuracy),
                    )
                )

            # Get next input sample.
            inputs = {"X": batch["encoded_image"].view(self.timestep, 1, 1, 28, 28)}
            if self.gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}
                batch["label"] = batch["label"].cuda()

            # Run the network on the input.
            network.run(inputs=inputs, time=self.time, input_time_dim=1)

            # Get spikes of output neurons.
            spikes = self.spikes["Z"].get("s").squeeze()

            # Compare ground truth label and prediction label.
            correct_pred = self.predict(batch["label"], spikes, correct_pred)

            self.test_spike.append(batch["label"])
            self.test_spike.append(spikes.sum(0))

            network.reset_state_variables()  # Reset state variables.

        acc = 100 * correct_pred / len(dataloader)
        if data_mode == "Train" or data_mode == "Validation":
            self.acc_history['train_acc'].append(acc)
        elif data_mode == "Test":
            self.acc_history['test_acc'].append(acc)

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

    def tryplotsss(self) -> None:
        wei = self.network.connections[("X", "Y")].w
        self.plotf.plot_weight_maps(wei)

    def predict(
        self,
        label: torch.Tensor,
        spikes: torch.Tensor,
        correct_pred: int,
    ) -> int:
        """
        Compare the ground truth label from input sample with prediction label
        from SL neurons' spikes.

        :param label: Label of the input sample.
        :param spikes: Binary tensor of shape ``(timestep, n_outpt)`` of 
            a layer's spiking activity.
        :param correct_pred: Number of correct predictions.
        :param return: Return number of correct predictions.
        """
        label = label[0]

        # Sum over time dimension (spike ordering doesn't matter).
        n_spikes = spikes.sum(0)

        # Get network prediction which are max of layer-wise firing rates.
        prediction = torch.sort(n_spikes, descending=True)[1][0]

        # Save "label" vs "prediction" for checking purpose.
        msg = "Ground truth: {}, Predict: {}".format(label, prediction)

        if label != prediction:
            self.wrong_pred.append(msg)
        else:
            correct_pred += 1
            self.right_pred.append(msg)

        return correct_pred

    def show_acc(self) -> None:
        """
        Show final accuracy of the network.
        """
        msg = []
        train_acc = test_acc = "N/A"

        if len(self.acc_history['train_acc']) != 0:
            train_acc = '%.2f' % np.mean(self.acc_history['train_acc'])
        if len(self.acc_history['test_acc']) != 0:
            test_acc = '%.2f' % np.mean(self.acc_history['test_acc'])

        train_msg = "Network train accuracy: " + train_acc
        test_msg = "Network test accuracy: " + test_acc

        msg.append(train_msg)
        msg.append(test_msg)

        msg_wrapper(msg, 2)

    def save_sl_spikes(self) -> None:
        """
        Save spike results for checking purpose.
        """
        file_path = os.path.join(self.results_path, "train_spike.txt")
        with open(file_path, 'w') as filehandle:
            filehandle.writelines("%s\n" % line for line in self.train_spike)

        file_path = os.path.join(self.results_path, "test_spike.txt")
        with open(file_path, 'w') as filehandle:
            filehandle.writelines("%s\n" % line for line in self.test_spike)

    def save_pred(self) -> None:
        """
        Save prediction results for checking purpose.
        """
        file_path = os.path.join(self.results_path, "right_pred.txt")
        with open(file_path, 'w') as filehandle:
            filehandle.writelines("%s\n" % line for line in self.right_pred)

        file_path = os.path.join(self.results_path, "wrong_pred.txt")
        with open(file_path, 'w') as filehandle:
            filehandle.writelines("%s\n" % line for line in self.wrong_pred)

    def tryplot(self) -> None:
        file_path = os.path.join(self.results_path, "init_exc.png")
        self.plotf.plot_weight_maps(self.exc_init_weight, file_path=file_path)

        file_path = os.path.join(self.results_path, "init_sl.png")
        self.plotf.plot_weight_maps(
            self.sl_init_weight, file_path=file_path,
            fig_shape=(4, 3), re_shape=(10, 10),
        )

        # weight = self.network.connections[("X", "Y")].w.detach().clone()
        file_path = os.path.join(self.results_path, "final_exc.png")
        self.plotf.plot_weight_maps(self.exc_final_weight, file_path=file_path)

        # weight = self.network.connections[("Y", "Z")].w.detach().clone()
        file_path = os.path.join(self.results_path, "final_sl.png")
        self.plotf.plot_weight_maps(
            self.sl_final_weight, file_path=file_path,
            fig_shape=(4, 3), re_shape=(10, 10),
        )

    def save_results(self) -> None:
        """
        Save trained network & plots of weight map.
        """
        # Save trained network.
        self.network.save(self.results_path + '/test_1_trained_network.pt')

        # Save plots.

        # if plot:
        #     # Alternative way:
        #     # plt.savefig(RESULTS_PATH + '/final.png')
        #     img = plot_weights(in_square_weights, im=in_weights_im).figure
        #     img.savefig(RESULTS_PATH + '/test_1_exc_weight.png')
        #     img2 = plot_weights(out_square_weights, im=out_weights_im).figure
        #     img2.savefig(RESULTS_PATH + '/test_1_sl_weight.png')
        #     imageio.mimwrite(RESULTS_PATH + '/test_1_exc_weight.gif', training_progress_images)
