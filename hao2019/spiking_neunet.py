from typing import List, Dict

import os
import torch
import numpy as np

from tqdm import tqdm

from bindsnet.encoding import poisson
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor

from .plot import Plot
from .utils import load_data, make_dirs, msg_wrapper, sample_from_class


class Spiking:
    """
    Class to maintain training and testing functions for spiking neural network (SNN).
    """

    def __init__(
        self,
        network: Network,
        results_path: str,
        dataset_name: str = "MNIST",
        seed: int = 0,
        batch_size: int = 1,
        n_epochs: int = 1,
        n_workers: int = -1,
        update_interval: int = 250,
        gif: bool = False,
        gpu: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Constructor for class Spiking.

        :param network:         Network model to use.
        :param results_path:    Path to save training & testing results.
        :param dataset_name:    Name of dataset to use.
        :param seed:            Seed for pseudorandom number generator (PRNG).
        :param batch_size:      Mini-batch size.
        :param n_epochs:        Number of epochs for training.
        :param n_workers:       Number of workers to use.
        :param update_interval: Interval to show network accuracy.
        :param gif:             Whether to create gif of weight maps.
        :param gpu:             Whether to use gpu.
        :param debug:           Whether to save files for debugging purpose.
        """
        self.network = network
        self.results_path = results_path
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_workers = n_workers
        self.update_interval = update_interval / batch_size
        self.gif = gif
        self.gpu = gpu
        self.debug = debug

        self.n_outpt = network.layers["Z"].n
        self.profile = {
            'method': network.method,
            'dataset_name': dataset_name,
            'n_train': None,
            'n_test': None,
        }

        self.time = network.time
        self.dt = network.dt
        self.timestep = int(self.time / self.dt)

        self.start_intensity_scale = 2
        self.start_intensity = 256.0 / 8.0 * self.start_intensity_scale

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        # Store results for checking purpose.
        self.sl_train_spike = []
        self.sl_test_spike = []
        self.right_pred = []
        self.wrong_pred = []

        # Store network accuracy.
        self.acc_history = {'train_acc': [], 'test_acc': []}
        self.store_pred = {}

        # Initialize plot class.
        self.visualize = Plot()

        # Save initial weights for plot.
        self.exc_init_weight = network.connections[("X", "Y")].w.detach().clone()
        self.sl_init_weight = network.connections[("Y", "Z")].w.detach().clone()

        # Determines number of workers to use.
        if n_workers == -1:
            self.n_workers = gpu * 4 * torch.cuda.device_count()

        # Sets max number of images to use for gif.
        if gif:
            self.n_gif_img = 35

        # Sets up Gpu use.
        if gpu:
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)

        # Directs network to GPU.
        if gpu:
            network.to("cuda")

        # Load train & test data.
        encoder = PoissonEncoder(time=self.time, dt=self.dt)
        self.train_dataset = load_data(dataset_name, encoder, True, self.start_intensity)
        self.test_dataset = load_data(dataset_name, encoder, False, self.start_intensity)

        # Set up monitors for spikes and voltages.
        spikes = {}
        for layer in set(network.layers):
            l = network.layers[layer]
            spikes[layer] = Monitor(l, state_vars=["s"], time=self.timestep)
            network.add_monitor(spikes[layer], name="%s_spikes" % layer)

        voltages = {}
        for layer in set(network.layers) - {"X"}:
            l = network.layers[layer]
            voltages[layer] = Monitor(l, state_vars=["v"], time=self.timestep)
            network.add_monitor(voltages[layer], name="%s_voltages" % layer)

        self.spikes = spikes
        self.voltages = voltages

    def train_network(self, lbyl_method: bool = False, **kwargs) -> None:
        """
        Train the spiking neural network.

        :param lbyl_method: Whether to use layer-by-layer training method.

        Keyword arguments:

        :param n_train: Number of samples of each class to use from dataset for training.
        :param n_test: Number of samples of each class to use from dataset for testing.
        :param bool infer_train: Whether to perform inference on train dataset for each epoch.
        :param bool infer_test: Whether to perform inference on test dataset for each epoch.
        :param bool shuffle_train: Whether to shuffle dataset for training.
        :param bool shuffle_test: Whether to shuffle dataset for testing.
        """
        # Parse keyword arguments.
        n_train = kwargs.get("n_train", None)
        n_test = kwargs.get("n_test", None)
        infer_train = kwargs.get("infer_train", False)
        infer_test = kwargs.get("infer_test", False)
        shuffle_train = kwargs.get("shuffle_train", False)
        shuffle_test = kwargs.get("shuffle_test", True)

        for epoch in range(self.n_epochs):
            print("Epoch progress: %d / %d" % (epoch, self.n_epochs))

            if not lbyl_method:
                self.train_network_sim(n_samples=n_train, shuffle=shuffle_train)
            else:
                self.train_network_lbyl(n_samples=n_train, shuffle=shuffle_train)

            # Perform a test on train dataset.
            if infer_train:
                self.test_network(n_samples=n_train, data_mode="train", shuffle=shuffle_test)
            # Perform a test on test dataset.
            if infer_test:
                self.test_network(n_samples=n_test, data_mode="test", shuffle=shuffle_test)

        print("Epoch progress: %d / %d" % (epoch + 1, self.n_epochs))

    def train_network_sim(self, n_samples: int = None, shuffle: bool = False) -> None:
        """
        Train the spiking neural network by using simultaneous training
        method from Hao's paper.

        :param n_samples: Number of samples of each class to use from dataset for training.
        :param shuffle: Whether to shuffle the dataset. Default to False.
        """
        print("Simultaneous training method.")
        self.profile['method'] = "Simultaneous"

        # Set train dataset as default dataset.
        dataset = self.train_dataset

        # Stratified sampling.
        if n_samples is not None and n_samples > 0:
            dataset = sample_from_class(dataset=dataset, n_samples=n_samples)
        self.profile['n_train'] = len(dataset)

        # Create a dataloader to iterate and batch data.
        dataloader = self.get_dataloader(dataset, shuffle=shuffle)

        network = self.network

        # Change training mode of network to True.
        network.train(True)

        # Determine the interval to plot weight map for gif.
        if self.gif:
            data_length = len(dataset) * self.n_epochs
            if data_length <= (self.n_gif_img * 2):
                gif_interval = 2
            else:
                gif_interval = int(data_length / self.n_gif_img)

        progress = tqdm(dataloader)
        for step, batch in enumerate(progress):
            # Plot a weight map image for gif.
            if self.gif and step % gif_interval == 0:
                exc_weight = network.connections[("X", "Y")].w
                self.visualize.plot_weight_maps(exc_weight, gif=self.gif)

            # Generate 0Hz or 200Hz Poisson rates for SL neurons in training mode.
            sl_label = torch.zeros(self.n_outpt)
            sl_label[batch["label"]] = 200
            sl_spike = poisson(datum=sl_label, time=self.time, dt=self.dt)

            # Get next input sample & SL neurons one-hot spikes.
            batch_input = batch["encoded_image"].view(
                self.timestep, self.batch_size, 1, 28, 28
            )
            inputs = {"X": batch_input}
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

            # Re-present input sample if less than five spikes.
            if exc_spike_count < 5:
                self.rerun_network(ori_image=batch["image"], clamp=clamp)

            if self.debug:
                self.sl_train_spike.append(batch["label"])
                sl_spike = self.spikes["Z"].get("s").squeeze().sum(0)
                self.sl_train_spike.append(sl_spike.cpu().numpy().tolist())
                self.sl_train_spike.append('')

            network.reset_state_variables()  # Reset state variables.

    def train_network_lbyl(self, n_samples: int = None, shuffle: bool = False) -> None:
        """
        Train the spiking neural network by using layer-by-layer training
        method from Hao's paper.

        :param n_samples: Number of samples of each class to use from dataset for training.
        :param shuffle: Whether to shuffle the dataset. Default to False.
        """
        print("Layer-by-layer training method.")
        self.profile['method'] = "Layer-by-layer"

        # Set train dataset as default dataset.
        dataset = self.train_dataset

        # Stratified sampling.
        if n_samples is not None and n_samples > 0:
            dataset = sample_from_class(dataset=dataset, n_samples=n_samples)
        self.profile['n_train'] = len(dataset)

        # Create a dataloader to iterate and batch data.
        dataloader = self.get_dataloader(dataset, shuffle=shuffle)

        network = self.network

        # Change training mode of network to True.
        network.train(True)

        # Determine the interval to plot weight map for gif.
        if self.gif:
            data_length = len(dataset) * self.n_epochs
            if data_length <= (self.n_gif_img * 2):
                gif_interval = 2
            else:
                gif_interval = int(data_length / self.n_gif_img)

        # Phase 1: Train only hidden (exc, inh) layer.
        # Phase 2: Train only SL layer.
        phase1 = True
        phase2 = False

        while phase1 or phase2:
            if phase1:
                print("\nBegin training for hidden layer.")
            elif phase2:
                print("\nBegin training for SL layer.")
                for layer in set(network.layers) - {"Z"}:
                    network.layers[layer].train(False)

            progress = tqdm(dataloader)
            for step, batch in enumerate(progress):
                # Plot a weight map image for gif.
                if phase1 and self.gif and step % gif_interval == 0:
                    exc_weight = network.connections[("X", "Y")].w
                    self.visualize.plot_weight_maps(exc_weight, gif=self.gif)

                # Get next input sample & SL neurons one-hot spikes.
                inputs = {"X": batch["encoded_image"].view(self.timestep, 1, 1, 28, 28)}

                # Do not generate spikes for SL neurons in phase 1.
                # Generate 0Hz or 200Hz Poisson rates for SL neurons in phase 2.
                if phase1:
                    clamp = {}
                elif phase2:
                    sl_label = torch.zeros(self.n_outpt)
                    sl_label[batch["label"]] = 200
                    sl_spike = poisson(datum=sl_label, time=self.time, dt=self.dt)
                    clamp = {"Z": sl_spike}

                if self.gpu:
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    clamp = {k: v.cuda() for k, v in clamp.items()}

                # Run the network on the input.
                network.run(inputs=inputs, time=self.time, input_time_dim=1, clamp=clamp)

                # Calculate number of spikes from excitatory neurons.
                exc_spike_count = self.spikes["Y"].get("s").squeeze().sum()

                # Re-present input sample if less than five spikes.
                if exc_spike_count < 5:
                    self.rerun_network(ori_image=batch["image"], clamp=clamp)

                if phase2 and self.debug:
                    self.sl_train_spike.append(batch["label"])
                    sl_spike = self.spikes["Z"].get("s").squeeze().sum(0)
                    self.sl_train_spike.append(sl_spike.cpu().numpy().tolist())
                    self.sl_train_spike.append('')

                network.reset_state_variables()  # Reset state variables.

            phase2 = phase1
            phase1 = False if phase1 else phase1

    def test_network(
        self, n_samples: int = None, data_mode: str = "test", shuffle: bool = True
    ) -> None:
        """
        Test the spiking neural network.

        :param n_samples: Number of samples of each class to use from dataset for testing.
        :param data_mode: Specifies (train / validation / test) dataset
            to use for testing.
        :param shuffle: Whether to shuffle the dataset. Default to True.
        """
        # Set test dataset as default dataset.
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

            # Set dataset based on mode.
            if data_mode == 'Train':
                dataset = self.train_dataset
            elif data_mode == 'Validation':
                dataset = self.validation_dataset

        # Stratified sampling.
        if n_samples is not None and n_samples > 0:
            dataset = sample_from_class(dataset=dataset, n_samples=n_samples)
        self.profile['n_test'] = len(dataset)

        # Create a dataloader for test data.
        dataloader = self.get_dataloader(dataset, shuffle=shuffle)

        network = self.network

        # Change training mode of network to False.
        network.train(False)

        accuracy = []
        self.store_pred = {}

        progress = tqdm(dataloader)
        for step, batch in enumerate(progress):
            # Calculate network accuracy at every update interval.
            if step % self.update_interval == 0 and step > 0:
                tmp_acc = 100 * sum(self.store_pred.values()) / step
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

            # Calculate number of spikes from SL neurons.
            sl_spike_count = self.spikes["Z"].get("s").squeeze().sum()

            # Re-present input sample if no spikes.
            if sl_spike_count <= 0:
                self.rerun_network_for_inference(ori_image=batch["image"])

            # Get spikes of output neurons.
            spikes = self.spikes["Z"].get("s").squeeze()

            # Compare ground truth label and prediction label.
            self.predict(batch["label"], spikes)

            network.reset_state_variables()  # Reset state variables.

        # Calculate accuracy.
        acc = 100 * sum(self.store_pred.values()) / len(dataloader)
        print("%s accuracy of current epoch: %.2f%%\n" % (data_mode, acc))

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
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.n_workers,
            pin_memory=self.gpu,
        )

        return dataloader

    def rerun_network(
        self,
        ori_image: torch.Tensor,
        clamp: Dict[str, torch.Tensor] = {},
    ) -> None:
        """
        Re-present the input sample with increased firing rate if excitatory neurons
        fire less than five spikes.
        NOTE: This function is for training.

        :param ori_image: Tensor of shape ``[batch_size, *input_shape]``
            of the original image's pixel intensity.
        :param clamp: Spikes to be clamped to SL neurons.
        """
        # Set intensity scale.
        intensity_scale = self.start_intensity_scale

        # Repeat until 5 spikes or above are obtained.
        while self.spikes["Y"].get("s").sum() < 5 and intensity_scale < 32:
            self.network.reset_state_variables()

            # Increase firing rate by 32Hz.
            intensity_scale += 1
            intensity = 256.0 / 8.0 * intensity_scale

            # Get new generated spikes.
            new_image = ori_image / self.start_intensity * intensity
            new_encoded_image = poisson(datum=new_image, time=self.time, dt=self.dt)

            inputs = {"X": new_encoded_image}
            if self.gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            self.network.run(inputs=inputs, time=self.time, input_time_dim=1, clamp=clamp)

    def rerun_network_for_inference(
        self,
        ori_image: torch.Tensor,
    ) -> None:
        """
        Re-present the input sample with increased firing rate until SL neurons
        respond with at least a spike during inference.
        NOTE: This function is for inference.

        :param ori_image: Tensor of shape ``[batch_size, *input_shape]``
            of the original image's pixel intensity.
        """
        # Set intensity scale.
        intensity_scale = self.start_intensity_scale

        # Repeat until 5 spikes or above are obtained.
        while self.spikes["Z"].get("s").sum() <= 0 and intensity_scale < 32:
            self.network.reset_state_variables()

            # Increase firing rate by 32Hz.
            intensity_scale += 1
            intensity = 256.0 / 8.0 * intensity_scale

            # Get new generated spikes.
            new_image = ori_image / self.start_intensity * intensity
            new_encoded_image = poisson(datum=new_image, time=self.time, dt=self.dt)

            inputs = {"X": new_encoded_image}
            if self.gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            self.network.run(inputs=inputs, time=self.time, input_time_dim=1)

    def predict(self, label: torch.Tensor, spikes: torch.Tensor) -> None:
        """
        Compare the ground truth label from input sample with prediction label
        from SL neurons' spikes.

        :param label: Label of the input sample.
        :param spikes: Binary tensor of shape ``(timestep, n_outpt)`` of 
            a layer's spiking activity.
        """
        label = label[0]

        # Sum over time dimension (spike ordering doesn't matter).
        n_spikes = spikes.sum(0)

        if n_spikes.sum() == 0:
            # Do not predict any of the class if no spikes are present.
            prediction = "No spikes"
        else:
            # Get network prediction which are max of layer-wise firing rates.
            prediction = torch.sort(n_spikes, descending=True)[1][0]

        if self.debug:
            # Save "label" vs "prediction" for checking purpose.
            msg = "Ground truth: {}, Predict: {}".format(label, prediction)

            self.sl_test_spike.append(msg)
            self.sl_test_spike.append(n_spikes.cpu().numpy().tolist())
            self.sl_test_spike.append('')

            if str(label) != str(prediction):
                self.wrong_pred.append(msg)
            else:
                self.right_pred.append(msg)

        if str(label) == str(prediction):
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.item()

            # Increment the correctly predicted class.
            self.store_pred[prediction] = self.store_pred.get(prediction, 0) + 1

    def calc_final_acc(self) -> str:
        """
        Calculate final average accuracies of the network.

        :return: Return train & test accuracies.
        """
        train_acc = test_acc = "N/A"

        if len(self.acc_history['train_acc']) != 0:
            train_acc = '%.2f' % np.mean(self.acc_history['train_acc']) + '%'
        if len(self.acc_history['test_acc']) != 0:
            test_acc = '%.2f' % np.mean(self.acc_history['test_acc']) + '%'

        return train_acc, test_acc

    def show_final_acc(self) -> None:
        """
        Show final average accuracies of the network.
        """
        msg = ["+ Network Average Accuracies +"]

        train_acc, test_acc = self.calc_final_acc()

        acc_msg = "Train accuracy: " + train_acc
        msg.append(acc_msg)
        acc_msg = "Test accuracy: " + test_acc
        msg.append(acc_msg)

        msg_wrapper(msg, 2)

    def write_file(self, content: List[str], file_name: str) -> None:
        """
        Save / write to file.

        :param content: Content to write to the file.
        :param file_name: Filename to use when saving.
        """
        # Setup save path.
        file_path = os.path.join(self.results_path, file_name)
        make_dirs(os.path.dirname(file_path))

        with open(file_path, 'w') as filehandle:
            filehandle.writelines("%s\n" % line for line in content)

    def save_sl_spike(self) -> None:
        """
        Save spike results for checking purpose.
        """
        # self.write_file(self.sl_train_spike, "debug/sl_train_spike.txt")
        self.write_file(self.sl_test_spike, "debug/sl_test_spike.txt")

    def save_pred(self) -> None:
        """
        Save prediction results for checking purpose.
        """
        self.write_file(self.right_pred, "debug/right_pred.txt")
        self.write_file(self.wrong_pred, "debug/wrong_pred.txt")

    def save_wmaps_plot(self, save_extension: str = 'png') -> None:
        """
        Save plots of neurons' initial weight maps and trained weight maps.

        :param save_extension: Filename extension for saving plot.
        """
        exc_conn = self.network.connections[("X", "Y")]
        sl_conn = self.network.connections[("Y", "Z")]

        file_name = "init_exc." + save_extension
        file_path = os.path.join(self.results_path, 'w_maps', file_name)
        self.visualize.plot_weight_maps(
            self.exc_init_weight,
            c_min=exc_conn.wmin,
            c_max=exc_conn.wmax,
            save=True,
            file_path=file_path,
        )

        file_name = "final_exc." + save_extension
        file_path = os.path.join(self.results_path, 'w_maps', file_name)
        self.visualize.plot_weight_maps(
            exc_conn.w,
            c_min=exc_conn.wmin,
            c_max=exc_conn.wmax,
            save=True,
            file_path=file_path,
        )

        file_name = "init_sl." + save_extension
        file_path = os.path.join(self.results_path, 'w_maps', file_name)
        self.visualize.plot_weight_maps(
            self.sl_init_weight,
            fig_shape=(4, 3),
            c_min=sl_conn.wmin,
            c_max=sl_conn.wmax,
            save=True,
            file_path=file_path,
        )

        file_name = "final_sl." + save_extension
        file_path = os.path.join(self.results_path, 'w_maps', file_name)
        self.visualize.plot_weight_maps(
            sl_conn.w,
            fig_shape=(4, 3),
            c_min=sl_conn.wmin,
            c_max=sl_conn.wmax,
            save=True,
            file_path=file_path,
        )

        file_name = "exc_overview." + save_extension
        file_path = os.path.join(self.results_path, 'w_maps', file_name)
        self.visualize.plot_weight_maps(
            exc_conn.w,
            c_min=exc_conn.wmin,
            c_max=exc_conn.wmax,
            overview=True,
            save=True,
            file_path=file_path,
        )

    def save_result(self) -> None:
        """
        Save trained network & accuracy results.
        """
        # Setup save path.
        make_dirs(self.results_path)

        # Save trained network.
        file_path = os.path.join(self.results_path, "trained_network.pt")
        self.network.save(file_path)

        # Save network profile, results & accuracy plots.
        self.save_network_details()
        self.save_acc()

        # Save weight maps' plots.
        self.save_wmaps_plot()

        # Save gif.
        if self.gif:
            # Plot the last weight map for gif.
            weight = self.network.connections[("X", "Y")].w
            self.visualize.plot_weight_maps(weight, gif=self.gif)

            file_path = os.path.join(self.results_path, "weight_maps.gif")
            self.visualize.save_wmaps_gif(file_path=file_path)

        # Save for checking purpose (debug).
        if self.debug:
            self.save_sl_spike()
            self.save_pred()

    def save_network_details(self) -> None:
        """
        Save network profile and experiment conditions.
        """
        train_acc, test_acc = self.calc_final_acc()

        file_path = os.path.join(self.results_path, "results.txt")
        with open(file_path, 'w') as f:
            f.write("# Network Architecture #\n\n")
            f.write("Number of neurons in layer:\n")
            f.write("    Input  -> {}\n".format(self.network.layers["X"].n))
            f.write("    Hidden -> {}\n".format(self.network.layers["Y"].n))
            f.write("    Output -> {}\n\n".format(self.network.layers["Z"].n))
            f.write("Spike presentation time : {} ms\n".format(self.time))
            f.write("Simulation time step    : {}\n\n".format(self.dt))
            f.write("Training method : {}\n".format(self.profile['method']))
            f.write("Dataset name    : {}\n".format(self.profile['dataset_name']))
            f.write("Minibatch size  : {}\n".format(self.batch_size))
            f.write("Number of epochs: {}\n\n".format(self.n_epochs))
            f.write("Number of data used:\n")
            f.write("    Training -> {}\n".format(self.profile['n_train']))
            f.write("    Testing  -> {}\n\n".format(self.profile['n_test']))
            f.write("Network's average accuracy:\n")
            f.write("    Train -> {}\n".format(train_acc))
            f.write("    Test  -> {}\n".format(test_acc))
            f.close()

    def save_acc(self) -> None:
        """
        Save network accuracy graph. Also write accuracy of each epoch to file.
        """
        if any(len(v) > 1 for v in self.acc_history.values()):
            file_path = os.path.join(self.results_path, "acc_graph.png")
            self.visualize.plot_accuracy(self.acc_history, file_path=file_path)

        for acc in self.acc_history:
            # Only save to text file when accuracy list is not empty.
            if len(self.acc_history[acc]) != 0:
                self.acc_history[acc].insert(0, "Accuracy of each epoch:")
                self.write_file(self.acc_history[acc], acc + ".txt")
