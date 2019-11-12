from typing import Tuple

import uuid
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)


class Plot:
    """
    Class for visualization by plot functions.
    """

    inpt_ims, inpt_axes = None, None
    spike_ims, spike_axes = None, None
    in_weights_im = None
    out_weights_im = None
    assigns_im = None
    perf_ax = None
    voltage_axes, voltage_ims = None, None
    weight_map_images = []

    # n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
    # n_sqrt2 = int(np.ceil(np.sqrt(n_outpt)))

    DPI: int = 300

    def plot_every_step(self) -> None:
        image = batch["image"].view(28, 28)
        inpt = inputs["X"].view(int(time/dt), 784).sum(0).view(28, 28)

        input_exc_weights = network.connections[("X", "Y")].w
        in_square_weights = get_square_weights(
            input_exc_weights.view(784, n_neurons), n_sqrt, 28
        )

        output_exc_weights = network.connections[("Y", "Z")].w
        out_square_weights = get_square_weights(
            output_exc_weights.view(n_neurons, n_outpt), n_sqrt2, 10
        )

        square_assignments = get_square_assignments(assignments, n_sqrt)
        spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
        voltages_ = {"Y": y_voltages}

        """ New version.
        # image = batch["image"][:, 0].view(28, 28)
        # inpt = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
        # spikes_ = {
        #         layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
        # }
        """

        # inpt_axes, inpt_ims = plot_input(
        #     image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
        # )
        # spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
        in_weights_im = plot_weights(in_square_weights, im=in_weights_im)
        out_weights_im = plot_weights(out_square_weights, im=out_weights_im)
        # assigns_im = plot_assignments(square_assignments, im=assigns_im)
        # perf_ax = plot_performance(accuracy, ax=perf_ax)
        # voltage_ims, voltage_axes = plot_voltages(
        #     voltages_, ims=voltage_ims, axes=voltage_axes, plot_type="line"
        # )

        plt.pause(1e-8)
        #plt.pause(0.5)

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
        :param file_path: File path (contains filename) to use when saving.
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
        :param file_path: File path (contains filename) to use when saving.
        """
        plt.savefig(file_path, dpi=dpi, bbox_inches='tight')

    def save_wmaps_gif(self, file_path: str = str(uuid.uuid4())) -> None:
        """
        Save gif of weight maps.

        :param file_path: File path (contains filename) to use when saving.
        """
        imageio.mimwrite(file_path, self.weight_map_images, duration=0.25)
