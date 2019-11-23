from typing import Tuple, Dict, List

import uuid
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

from bindsnet.network.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_performance,
    plot_voltages,
)


class Plot:
    """
    Class for visualization by plot functions.

    Special thanks to Araki Hiroshi for his excellent teaching and help!!!
    Few of the plotting functions below were borrowed from his code in
    `<https://github.com/HiroshiARAKI/snnlibpy>`.
    """

    # Turn the interactive mode off.
    plt.ioff()

    DPI: int = 300
    weight_map_images = []

    def plot_weight_maps(
        self,
        weights: torch.Tensor,
        fig_shape: Tuple[int, int] = (3, 3),
        c_min: float = 0.0,
        c_max: float = 1.0,
        gif: bool = False,
        save: bool = False,
        file_path: str = str(uuid.uuid4()),
    ) -> None:
        """
        Plot weight maps of a layer's neurons with the shape of [fig_shape].

        :param weights: Weight matrix of ``Connection`` object.
        :param fig_shape: Horizontal, vertical figure shape for plot.
        :param c_min: Lower bound of the range that the colormap covers.
        :param c_max: Upper bound of the range that the colormap covers.
        :param gif: Save plot of weight maps for gif.
        :param save: Whether to save the plot's figure.
        :param file_path: Path (contains filename) to use when saving the object.
        """
        # Turn the interactive mode off if just for saving.
        if save or gif:
            plt.ioff()

        # Number of neurons from front layer.
        n_pre_neu = len(weights)

        # Calculate the perfect square which is closest to the number of neurons.
        size = int(np.sqrt(n_pre_neu))
        # max_size = 30
        # if restrict and size > max_size:
        #     size = max_size
        sq_size = size ** 2
        sq_shape = (size, size)

        # Convert torch Tensor to numpy Array and transpose it.
        weights = weights.cpu().numpy().T
        # weights = weights.detach().clone().cpu().numpy().T

        # Number of neurons from current layer.
        n_post_neu = len(weights)

        # Create figure of shape (m, n).
        fig, axes = plt.subplots(ncols=fig_shape[0], nrows=fig_shape[1])
        fig.subplots_adjust(right=0.8, hspace=0.28)

        index = 0
        for cols in axes:
            for ax in cols:
                if index >= n_post_neu:
                    ax.axis('off')
                    continue

                # Slice the array of weight map to fit perfect square's shape.
                if len(weights[index]) > sq_size:
                    tmp_weights = weights[index][:sq_size]
                else:
                    tmp_weights = weights[index]

                tmp_weights = tmp_weights.reshape(sq_shape)

                im = ax.imshow(tmp_weights, cmap='BuPu', vmin=c_min, vmax=c_max)
                ax.set_title('Map ({})'.format(index))
                ax.tick_params(
                    labelbottom=False,
                    labelleft=False,
                    labelright=False,
                    labeltop=False,
                    bottom=False,
                    left=False,
                    right=False,
                    top=False
                )
                index += 1

        # cbar_ax = fig.add_axes([0.85, 0.11, 0.03, 0.77])
        cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        if gif:
            self.wmaps_for_gif()

        if save:
            self.save_plot(file_path=file_path)

        plt.close()

    def plot_acc(
        self,
        acc_history: Dict[str, List[float]],
        file_path: str = str(uuid.uuid4()),
    ) -> None:
        """
        Plot and save network accuracy graph.

        :param acc_history: List of train and test accuracy of each epoch.
        :param file_path: Path (contains filename) to use when saving the object.
        """
        epochs = max([len(acc_history[acc]) for acc in acc_history])
        epochs = np.arange(1, epochs+1)

        for acc in acc_history:
            if len(acc_history[acc]) != 0:
                plt.plot(epochs, acc_history[acc], label=acc, marker='x')

        plt.xticks(epochs)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        self.save_plot(file_path=file_path)
        plt.close()

    def wmaps_for_gif(self) -> None:
        """
        Store weight map images for gif.
        """
        # Convert figure to numpy array.
        fig = plt.gcf()
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        self.weight_map_images.append(data)

    def save_plot(self, dpi: int = DPI, file_path: str = str(uuid.uuid4())) -> None:
        """
        Save plot.

        :param dpi: Output resolution to use when saving image.
        :param file_path: Path (contains filename) to use when saving the object.
        """
        plt.savefig(file_path, dpi=dpi, bbox_inches='tight')

    def save_wmaps_gif(self, file_path: str = str(uuid.uuid4())) -> None:
        """
        Save gif of weight maps.

        :param file_path: Path (contains filename) to use when saving the object.
        """
        if self.weight_map_images:
            imageio.mimwrite(file_path, self.weight_map_images)

    def plot_every_step(
        self,
        batch: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor],
        spikes: Monitor,
        voltages: Monitor,
        accuracy: float = None,
    ) -> None:
        """
        Visualize network's training process.
        *** This function is currently broken and unusable. ***

        :param batch: Current batch from dataset.
        :param inputs: Current inputs from batch.
        :param spikes: Spike monitor.
        :param voltages: Voltage monitor.
        :param accuracy: Network accuracy.
        """
        inpt_ims, inpt_axes = None, None
        spike_ims, spike_axes = None, None
        weights_im = None
        assigns_im = None
        perf_ax = None
        voltage_axes, voltage_ims = None, None

        n_inpt = self.network.n_inpt
        n_neurons = self.network.n_neurons
        n_outpt = self.network.n_outpt
        inpt_sqrt = int(np.ceil(np.sqrt(n_inpt)))
        neu_sqrt = int(np.ceil(np.sqrt(n_neurons)))
        outpt_sqrt = int(np.ceil(np.sqrt(n_outpt)))
        inpt_view = (inpt_sqrt, inpt_sqrt)

        image = batch["image"].view(inpt_view)
        inpt = inputs["X"].view(self.timestep, n_inpt).sum(0).view(inpt_view)

        input_exc_weights = self.network.connections[("X", "Y")].w
        in_square_weights = get_square_weights(
            input_exc_weights.view(n_inpt, n_neurons), neu_sqrt, inpt_sqrt
        )

        output_exc_weights = self.network.connections[("Y", "Z")].w
        out_square_weights = get_square_weights(
            output_exc_weights.view(n_neurons, n_outpt), outpt_sqrt, neu_sqrt
        )

        spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
        voltages_ = {layer: voltages[layer].get("v") for layer in voltages}

        """ For mini-batch.
        # image = batch["image"][:, 0].view(28, 28)
        # inpt = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
        # spikes_ = {
        #         layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
        # }
        """

        inpt_axes, inpt_ims = plot_input(
            image, inpt, label=batch["label"], axes=self.inpt_axes, ims=self.inpt_ims
        )
        spike_ims, spike_axes = plot_spikes(
            spikes_, ims=self.spike_ims, axes=self.spike_axes
        )
        in_weights_im = plot_weights(in_square_weights, im=self.in_weights_im)
        out_weights_im = plot_weights(out_square_weights, im=self.out_weights_im)
        if accuracy is not None:
            perf_ax = plot_performance(accuracy, ax=self.perf_ax)
        voltage_ims, voltage_axes = plot_voltages(
            voltages_, ims=self.voltage_ims, axes=self.voltage_axes, plot_type="line"
        )

        plt.pause(1e-8)
