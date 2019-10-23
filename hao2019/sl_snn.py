import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imageio

from tqdm import tqdm

from time import time as t

from bindsnet.encoding import PoissonEncoder
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)

from model import HaoAndHuang2019
from utils import load_data, sl_poisson


# Define dataset and number of input / output neurons.
dataset_name = 'MNIST'
n_inpt = 784
n_outpt = 10
# model_name = 'hao_2019'
results_path = os.path.join('results', dataset_name.lower())
#gif_path = os.path.join('results', dataset_name.lower(), 'gif')
# torch.set_printoptions(profile="full")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=10)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--norm_scale", type=float, default=0.1)
parser.add_argument("--theta_plus", type=float, default=0.07)
parser.add_argument("--time", type=int, default=350)
parser.add_argument("--dt", type=int, default=0.5)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(train=True, plot=False, gpu=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
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

# Sets up Gpu use
if gpu:
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
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

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load data.
encoder = PoissonEncoder(time=time, dt=dt)
dataset = load_data(dataset_name, encoder, intensity)

# Setup directories within path.
for path in [results_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons)

# Neuron assignments and spike proportions.
n_classes = n_outpt
assignments = -torch.ones(n_neurons)
proportions = torch.zeros(n_neurons, n_classes)
rates = torch.zeros(n_neurons, n_classes)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory layer.
exc_voltage_monitor = Monitor(network.layers["Y"], ["v"], time=time)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
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

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(time/dt), 1, 1, 28, 28)}

        # Generate 0Hz or 200Hz Poisson rates for SL neurons in training mode.
        if train:
            sl_label = torch.zeros(n_outpt).byte()
            sl_label[batch["label"]] = 1.0
            sl_spike = sl_poisson(datum=sl_label, time=time, dt=dt)
            
            clamp = {"Z": sl_spike}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
            clamp = {k: v.cuda() for k, v in clamp.items()}

        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        labels.append(batch["label"])
        
        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1, clamp=clamp)

        # Re-present the input sample with increased firing rate
        # if excitatory neurons fire less than five spikes.
        exc_spike = spikes["Y"].get("s").squeeze()
        exc_spike_count = torch.sum(torch.sum(exc_spike, dim=0), dim=0)
        while exc_spike_count < 5:
            network.reset_state_variables()
            #TODO increase firing rate?
            print("MOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOFO")
            network.run(inputs=inputs, time=time, input_time_dim=1, clamp=clamp)

            exc_spike = spikes["Y"].get("s").squeeze()
            exc_spike_count = torch.sum(torch.sum(exc_spike, dim=0), dim=0)

        # Get voltage recording.
        # exc_voltages = exc_voltage_monitor.get("v")
        y_voltages = voltages["Y"].get("v")
        #inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[step % update_interval] = spikes["Y"].get("s").squeeze()

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"].view(28, 28)
            inpt = inputs["X"].view(int(time/dt), 784).sum(0).view(28, 28)
            input_exc_weights = network.connections[("X", "Y")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            voltages_ = {"Y": y_voltages}

            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            # assigns_im = plot_assignments(square_assignments, im=assigns_im)
            # perf_ax = plot_performance(accuracy, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(
                voltages_, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            plt.pause(1e-8)
            #plt.pause(0.5)

            # # Create gif from plot images.
            # fig = weights_im.figure
            # # Convert figure to numpy array.
            # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # training_progress_images.append(data)

        if step > 20:
            #pass
            break

        network.reset_state_variables()  # Reset state variables.

# Save final network & plots.
#network.save(os.path.join(results_path, 'network.pt'))
if plot:
    # Alternative way:
    #plt.savefig(results_path + '/final.png')
    img = plot_weights(square_weights, im=weights_im).figure
    img.savefig(results_path + '/test_final_weight.png')
    # imageio.mimwrite(results_path + '/exc_weight.gif', training_progress_images)

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")
