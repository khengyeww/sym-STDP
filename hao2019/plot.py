import numpy as np
import matplotlib.pyplot as plt
import imageio

from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    Class for plot functions.
    """

    inpt_ims, inpt_axes = None, None
    spike_ims, spike_axes = None, None
    in_weights_im = None
    out_weights_im = None
    assigns_im = None
    perf_ax = None
    voltage_axes, voltage_ims = None, None
    training_progress_images = []

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

        # image = batch["image"][:, 0].view(28, 28)
        # inpt = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
        # spikes_ = {
        #         layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
        # }

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

    def weights_in_gif(self) -> None:
        """
        Create gif from weight images.
        """

        # Convert figure to numpy array.
        fig = in_weights_im.figure
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        self.training_progress_images.append(data)

    def plot_weight_maps(
        self, weight,#: torch.Tensor,
        file_path: str, fig_shape: tuple = (3, 3), re_shape: tuple = (28, 28),
        dpi: int = DPI, c_max: float = 1.0, c_min: float = 0.0):
        """
        Plot weight maps of output connection with the shape of [f_shape].
        :param f_shape:
        :param file_name:
        :param dpi:
        :param c_max:
        :param c_min:
        :param save:
        :return:
        """
        # Turn the interactive mode off.
        plt.ioff()

        n_pre_neu = len(weight)

        # Convert to N-dimensional array (ndarray) and transpose it.
        weight = weight.cpu().numpy().T
        # weight = weight.detach().clone().cpu().numpy().T
        n_post_neu = len(weight)

        # Create figure of shape (m, n).
        fig, axes = plt.subplots(ncols=fig_shape[0], nrows=fig_shape[1])
        fig.subplots_adjust(right=0.8, hspace=0.28)

        index = 0
        # im = None
        for cols in axes:
            for ax in cols:
                if index >= n_post_neu:
                    ax.axis('off')
                    continue

                # Reshape to (28, 28).
                tmp_weight = weight[index].reshape(re_shape)

                im = ax.imshow(tmp_weight, cmap='BuPu', vmax=c_max, vmin=c_min)
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

        plt.savefig(file_path, dpi=dpi)
