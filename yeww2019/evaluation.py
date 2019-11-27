"""
Example of loading trained network model for inference or further training.
"""

import os
import argparse
import time as date

from time import time as t

from bindsnet.network.network import load

from spiking_neunet import Spiking
from utils import msg_wrapper, make_dirs


dataset_name = 'MNIST'  # Dataset to use.
file_name = "results/trained_network.pt"  # Path to trained network.

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=5)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--update_interval", type=int, default=2000)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(gpu=False)

args = parser.parse_args()

seed = args.seed
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_workers = args.n_workers
update_interval = args.update_interval
gpu = args.gpu

# if n_train is not None:
#     assert (n_train > 0), "Samples for training must be greater than 0"
if n_test is not None:
    assert (n_test > 0), "Samples for testing must be greater than 0"

if gpu:
    device = "gpu"
else:
    device = "cpu"

# Setup pathnames for saving files.
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
datetime = date.strftime("%Y%m%d-%H%M%S")
data_name = dataset_name.lower()
epoch_num = 'epoch-' + str(n_epochs)
data_num = 'test-' + str(n_test)
DIR_NAME = "-EVALUATION-" + datetime + '_' + data_name + '_' + epoch_num + '_' + data_num
RESULTS_PATH = os.path.join(ROOT_PATH, 'results', DIR_NAME)

# Load trained network for evaluation.
network = load(file_name=file_name, map_location=device)

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
    gpu=gpu,
)

# ------------------------------------------------------------------------------- #

# Test the network.
msg = ["--- Begin testing. ---"]
msg_wrapper(msg, 1)
start_test = t()

# snn.test_network(n_test, data_mode='train', shuffle=True)
snn.test_network(n_test, data_mode='test', shuffle=True)

print("Testing complete. (%.4f minutes)\n" % ((t() - start_test) / 60))

# ------------------------------------------------------------------------------- #

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
