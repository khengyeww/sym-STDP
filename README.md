# Hao's sym-STDP-SNN model

Near-replication of Hao's SNN model from [this paper](https://www.sciencedirect.com/science/article/pii/S0893608019302680)
using BindsNET.

**Note:**
***\*\*\* Updated on 2020/02/04 \*\*\**** 

Original model (`v1`, `v2` models) was not able to replicate the results mentioned in the paper.
Created a `v3` model which achieves similar results from the paper by
modifying the weight normalization between `hidden layer -- SL layer`.

## Requirements
 - Ubuntu 16.04
 - Python 3.6
 - BindsNET 0.2.4
 
N.B. Versions mentioned above are verified to work together. Other versions might not be compatible.

## Setup

Head over to [BindsNET](https://github.com/Hananel-Hazan/bindsnet) and follow their instruction for environment setup.

Alternatively, install the required packages using requirements.txt provided:

```
pip install -r requirements.txt
```

## Run

Simply run the following command (both training and inference process included in the file):

```
python main_snn.py
```

Example of passing optional command-line arguments:

```bash
# (--n_neurons [int]): to set number of excitatory, inhibitory neurons
# (--gpu): to use GPU backend for calculation
python main_snn.py --n_neurons 400 --gpu
```

Run the script with the `--help` or `-h` flag for more information.

## Citation

 - **BindsNET**

	All of the SNN models in this project were implemented using BindsNET:

	Link to [GitHub](https://github.com/Hananel-Hazan/bindsnet).

	```
	@ARTICLE{10.3389/fninf.2018.00089,
		AUTHOR={Hazan, Hananel and Saunders, Daniel J. and Khan, Hassaan and Patel, Devdhar and Sanghavi, Darpan T. and Siegelmann, Hava T. and Kozma, Robert},
		TITLE={BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python},
		JOURNAL={Frontiers in Neuroinformatics},
		VOLUME={12},
		PAGES={89},
		YEAR={2018},
		URL={https://www.frontiersin.org/article/10.3389/fninf.2018.00089},
		DOI={10.3389/fninf.2018.00089},
		ISSN={1662-5196},
	}
	```

- **sym-STDP-SNN**

	Some of the parameters' value that are not mentioned in the paper were obtained from Hao's original code.

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

