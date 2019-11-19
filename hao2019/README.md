# Hao's sym-STDP-SNN model

Near-replication of Hao's SNN model from [this paper](https://www.sciencedirect.com/science/article/pii/S0893608019302680)
using BindsNET.

## Run

Simply run the following command (both training and inference process included in the file):

```
python main_snn.py
```

Example of passing optional command-line arguments:

```bash
# --n_neurons [int] to set number of excitatory, inhibitory neurons
# --gpu to use GPU backend for calculation
python main_snn.py --n_neurons 400 --gpu
```

Run the script with the `--help` or `-h` flag for more information.

## Citation

 - **sym-STDP-SNN**

	Some of the variables' value that are not mentioned in the paper were obtained from Hao's original code.

	Link to [GitHub](https://github.com/haoyz/sym-STDP-SNN).

	```
	@ARTICLE{hao2019biologically,
		AUTHOR={Hao, Yunzhe and Huang, Xuhui and Dong, Meng and Xu, Bo},
		TITLE={A biologically plausible supervised learning method for spiking neural networks using the symmetric STDP rule},
		JOURNAL={Neural Networks},
		VOLUME={121},
		PAGES={387-395},
		YEAR={2020},
		PUBLISHER={Elsevier},
		URL={https://www.sciencedirect.com/science/article/pii/S0893608019302680},
		DOI={10.1016/j.neunet.2019.09.007},
	}
	```
