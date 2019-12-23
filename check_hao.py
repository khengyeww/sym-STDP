"""
This file is used to verify test accuracy by using trained weights
provided by Hao from `https://github.com/haoyz/sym-STDP-SNN`_.
"""

import os
import torch
import argparse
import time as date
import numpy as np

from time import time as t

from hao2019.model import HaoAndHuang2019, HaoAndHuang2019v2
from hao2019.spiking_neunet import Spiking
from hao2019.utils import msg_wrapper


# Define dataset and number of input / output neurons.
dataset_name = 'MNIST'
n_inpt = 784
n_outpt = 10

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_train", type=int, default=None)
parser.add_argument("--n_test", type=int, default=None)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--norm_scale", type=float, default=0.1)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=350)
parser.add_argument("--dt", type=float, default=0.5)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=5000)
parser.add_argument("--lbyl", dest="lbyl", action="store_true")
parser.add_argument("--gif", dest="gif", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(lbyl=False, dynamic=False, gif=False, gpu=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_train = args.n_train
n_test = args.n_test
n_workers = args.n_workers
norm_scale = args.norm_scale
theta_plus = args.theta_plus
time = args.time
dt = args.dt
progress_interval = args.progress_interval
update_interval = args.update_interval
lbyl = args.lbyl
gif = args.gif
gpu = args.gpu

if n_train is not None:
    assert (n_train > 0), "Samples for training must be greater than 0"
if n_test is not None:
    assert (n_test > 0), "Samples for testing must be greater than 0"

# Setup pathnames for saving files.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
datetime = date.strftime("%Y%m%d-%H%M%S")
data_name = dataset_name.lower() + '-' + str(n_neurons)
epoch_num = 'epoch-' + str(n_epochs)
data_num = str(n_train) + ',' + str(n_test)
DIR_NAME = datetime + '_' + data_name + '_' + epoch_num + '_' + data_num
RESULTS_PATH = os.path.join(ROOT_PATH, 'results/hao_result', DIR_NAME)

# Build network model.
network = HaoAndHuang2019v2(
    n_inpt=n_inpt,
    n_outpt=n_outpt,
    n_neurons=n_neurons,
    time=time,
    dt=dt,
    norm_scale=norm_scale,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
    method=lbyl,
)

# Setup network for training & testing.
snn = Spiking(
    network=network,
    results_path=RESULTS_PATH,
    dataset_name=dataset_name,
    seed=seed,
    batch_size=batch_size,
    n_epochs=n_epochs,
    n_workers=n_workers,
    update_interval=update_interval,
    gif=gif,
    gpu=gpu,
)

trained = [
    'mnist_{}/gP2E'.format(n_neurons),
    'mnist_{}/gE2C'.format(n_neurons),
    'mnist_{}/theta'.format(n_neurons)
]
new_params = []
for file in trained:
    with open(file) as f:
        params = []
        for line in f:
            line = line.split()  # Skip blank lines.
            if line:
                line = [i for i in line]
                params.append(line)

    params = torch.nn.Parameter(torch.from_numpy(np.asarray(params, dtype=np.float32)))
    params.requires_grad = False
    new_params.append(params)

network.connections[('X', 'Y')].w = new_params[0]
network.connections[('Y', 'Z')].w = new_params[1]
network.layers['Y'].theta = new_params[2].squeeze(0)

# print(network.connections[('X', 'Y')].w.shape)
# print(network.connections[('Y', 'Z')].w.shape)
# print(network.layers['Y'].theta.shape)

# ------------------------------------------------------------------------------- #

"""
    ### Testing Session ###
"""
# Test the network.
msg = ["--- Begin testing. ---"]
msg_wrapper(msg, 1)
start_test = t()

snn.test_network(n_test, data_mode='test', shuffle=False)

print("Testing complete. (%.4f minutes)\n" % ((t() - start_test) / 60))

# ------------------------------------------------------------------------------- #

# Print final train & test accuracy.
snn.show_final_acc()

print("\nSaving network & results... ...\n")
snn.save_result()  # Save network & results.
snn.save_wmaps_plot()  # Save weight maps' plots.

# Save for checking purpose.
snn.save_sl_spike()
snn.save_pred()
print(" ... ...done!\n")
