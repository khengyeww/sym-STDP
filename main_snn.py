import os
import torch
import argparse
import time as date

from time import time as t

from hao2019 import (
    HaoAndHuang2019,
    HaoAndHuang2019v2,
    Spiking,
    msg_wrapper,
    make_dirs,
)


# Define dataset and number of input / output neurons.
dataset_name = 'MNIST'
n_inpt = 784
n_outpt = 10
# model_name = 'hao_2019'

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
parser.add_argument("--update_interval", type=int, default=2500)
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
RESULTS_PATH = os.path.join(ROOT_PATH, 'results', DIR_NAME)
# paths = [RESULTS_PATH]
# torch.set_printoptions(profile="full")

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

# ------------------------------------------------------------------------------- #

"""
    ### Training Session ###
"""
# Train the network.
msg = ["### Begin training. ###"]
msg_wrapper(msg, 1)
start = t()

for epoch in range(n_epochs):
    if epoch != 0:
        msg = ["+++ Resume training. +++"]
        msg_wrapper(msg, 1)

    if epoch % progress_interval == 0:
        print(
            "Progress: %d / %d (%.4f minutes)" % (epoch, n_epochs, ((t() - start) / 60))
        )
        start = t()

    if not lbyl:
        # Decide number of samples to use for training.
        # Default to all samples.
        snn.train_network(n_train)
    else:
        snn.train_network_lbyl(n_train)

    # ------------------------------------------------------------------------------- #

    """
        ### Testing Session ###
    """
    # Test the network.
    msg = ["--- Begin testing. ---"]
    msg_wrapper(msg, 1)
    start_test = t()

    # Decide number of samples & dataset to use for testing.
    # Default to all samples & test data mode (using test dataset).
    snn.test_network(n_train, data_mode='train')
    snn.test_network(n_test, data_mode='test')

    print("Testing complete. (%.4f minutes)\n" % ((t() - start_test) / 60))

    # ------------------------------------------------------------------------------- #

print("Progress: %d / %d (%.4f minutes)" % (epoch + 1, n_epochs, ((t() - start) / 60)))
print("Training complete.\n")

# ------------------------------------------------------------------------------- #

# Print final train & test accuracy.
snn.show_final_acc()

print("\nSaving network & results... ...\n")
# Setup directories within path.
make_dirs(RESULTS_PATH)

# Save network & results.
snn.save_result()
# Save weight maps' plots.
snn.save_wmaps_plot()

# Save for checking purpose.
snn.save_sl_spike()
snn.save_pred()
print(" ... ...done!\n")
