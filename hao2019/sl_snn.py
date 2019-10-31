import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio

from tqdm import tqdm

import time as date
from time import time as t

from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)

from model import HaoAndHuang2019
from spiking_neunet import Spiking


# Define dataset and number of input / output neurons.
dataset_name = 'MNIST'
n_inpt = 784
n_outpt = 10
# model_name = 'hao_2019'

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--norm_scale", type=float, default=0.1)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=350)
parser.add_argument("--dt", type=float, default=0.5)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=3)#250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false") #TODO
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(train=True, plot=False, gpu=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_workers = args.n_workers
exc = args.exc
inh = args.inh
norm_scale = args.norm_scale
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

datetime = date.strftime("%Y%m%d-%H%M%S")
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DIR_NAME = dataset_name.lower() + '-' + str(n_neurons) + '_' + datetime
RESULTS_PATH = os.path.join(ROOT_PATH, 'results', DIR_NAME)
# torch.set_printoptions(profile="full")

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
n_sqrt2 = int(np.ceil(np.sqrt(n_outpt)))
start_intensity = intensity

# Build network.
network = HaoAndHuang2019(
    n_inpt=n_inpt,
    n_outpt=n_outpt,
    n_neurons=n_neurons,
    inh=inh,
    dt=dt,
    norm_scale=norm_scale,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

snn = Spiking(
    network = network,
    n_outpt = n_outpt,
    dataset_name = dataset_name,
    seed = seed,
    n_workers = n_workers,
    time = time,
    dt = dt,
    intensity = intensity,
    update_interval = update_interval,
    gpu = gpu,
)

# Setup directories within path.
for path in [RESULTS_PATH]:
    if not os.path.isdir(path):
        os.makedirs(path)
    # Alternative way:
    # os.makedirs(path, exist_ok=True)

"""
    ### Training Session ###
""
inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
in_weights_im = None
out_weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None
training_progress_images = []

# Train the network.
print("\nBegin training.\n")
start = t()
clamp = {}

for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data.
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(time/dt), 1, 1, 28, 28)}

        # Generate 0Hz or 200Hz Poisson rates for SL neurons in training mode.
        if train:
            sl_label = torch.zeros(n_outpt).byte()
            sl_label[batch["label"]] = True
            sl_spike = sl_poisson(datum=sl_label, time=time, dt=dt)

            clamp = {"Z": sl_spike}

        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            clamp = {k: v.cuda() for k, v in clamp.items()}

        labels.append(batch["label"])
        
        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1, clamp=clamp)

        # torch.set_printoptions(profile="full")
        # print("checkpoi1")
        # y_spike = spikes["Y"].get("s").squeeze()
        # for timest, neuron in enumerate(y_spike):
        #     if neuron.sum() > 0:
        #         print(timest)
        #         print(neuron)
        # print(y_spike.sum())
        # print("checkpoi2")
        # z_spike = spikes["Z"].get("s")
        # for timest, neuron in enumerate(z_spike):
        #     if neuron.sum() > 0:
        #         print(timest)
        #         print(batch["label"])
        #         print(neuron)
        # print(z_spike.sum())

        #TODO
        i = 0
        # Re-present the input sample with increased firing rate
        # if excitatory neurons fire less than five spikes.
        exc_spike = spikes["Y"].get("s").squeeze()
        exc_spike_count = exc_spike.sum()
        # Alternative way:
        #exc_spike_count = torch.sum(torch.sum(exc_spike, dim=0), dim=0)
        while exc_spike_count < 5:
            i += 1
            network.reset_state_variables()
            #TODO increase firing rate?
            print("MOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOFO in line225")
            network.run(inputs=inputs, time=time, input_time_dim=1, clamp=clamp)

            exc_spike_count = spikes["Y"].get("s").squeeze().sum()

            if i > 5:
                break

        # Get voltage recording.
        # exc_voltages = exc_voltage_monitor.get("v")
        y_voltages = voltages["Y"].get("v")
        #inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[step % update_interval] = spikes["Z"].get("s").squeeze()

        # Optionally plot various simulation information.
        if plot:
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

            # Create gif from plot images.
            fig = in_weights_im.figure
            # Convert figure to numpy array.
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            training_progress_images.append(data)

        # if step > 5:
        #     #pass
        #     break

        network.reset_state_variables()  # Reset state variables.

# Save final network & plots.
network.save(os.path.join(RESULTS_PATH, 'test_1_network.pt'))
if plot:
    # Alternative way:
    #plt.savefig(RESULTS_PATH + '/final.png')
    img = plot_weights(in_square_weights, im=in_weights_im).figure
    img.savefig(RESULTS_PATH + '/test_1_exc_weight.png')
    img2 = plot_weights(out_square_weights, im=out_weights_im).figure
    img2.savefig(RESULTS_PATH + '/test_1_sl_weight.png')
    imageio.mimwrite(RESULTS_PATH + '/test_1_exc_weight.gif', training_progress_images)

print("Progress: %d / %d (%.4f minutes)" % (epoch + 1, n_epochs, ((t() - start) / 60)))
print("Training complete.\n")
"""
"""
    ### Testing Session ###
"""
# Test the network.
print("\nBegin testing.\n")
start = t()

snn.test_network(1)

print("Testing complete. (%.4f minutes)\n" % ((t() - start) / 60))

if len(snn.test_acc_history) != 0:
    print("Network test accuracy: %.2f\n" % np.mean(snn.test_acc_history))