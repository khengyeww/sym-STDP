import torch
import numpy as np

from tqdm import tqdm

from bindsnet.network.network import Network

from utils import prediction


def test_network(
    network: Network,
    dataset: torch.utils.data.Dataset,
    spikes: {},
    spike_record: torch.Tensor,
    n_workers: int,
    time: int,
    dt: float = 0.5,
    update_interval: int = 250,
    gpu: bool = False,
) -> None:
    """
    Testing network with test data.

    :param network: Network model to use.
    :param dataset: Dataset to use.
    :param spikes: Monitor for spikes.
    :param spike_record: Tensor of shape ``[update_interval, time, n_outpt]``
        for recording spikes during the simulation.
    :param n_workers: Number of workers to use.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :param update_interval: Interval to compute network accuracy.
    :param gpu: Whether to use gpu.
    """

    # Create a dataloader to iterate and batch data.
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    # Change learning mode of each layer's nodes to false.
    for l in network.layers:
        network.layers[l].learning = False

    labels = []
    accuracy = []

    for step, batch in enumerate(tqdm(dataloader)):
        # Update network accuracy at every interval.
        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels)

            # Get network predictions.
            test_pred = prediction(spikes=spike_record)

            # Compute network accuracy.
            accuracy.append(
                100
                * torch.sum(label_tensor.long() == test_pred).item()
                / len(label_tensor)
            )

            print(
                "\nTest accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy[-1],
                    np.mean(accuracy),
                    np.max(accuracy),
                )
            )

            labels = []

        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(time/dt), 1, 1, 28, 28)}

        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Save label of each sample.
        labels.append(batch["label"])
        # Add to spikes recording.
        spike_record[step % update_interval] = spikes["Z"].get("s").squeeze()

        network.reset_state_variables()  # Reset state variables.
