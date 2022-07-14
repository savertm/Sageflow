# Sageflow


This repo contains the PyTorch implementation of our NeurIPS'21 paper, [Sageflow: Robust Federated Learning against Both Stragglers and Adversaries at NeurIPS'21](https://proceedings.neurips.cc/paper/2021/hash/076a8133735eb5d7552dc195b125a454-Abstract.html)

**Abstract:** While federated learning (FL) allows efficient model training with local data at edge devices, among major issues still to be resolved are: slow devices known as stragglers and malicious attacks launched by adversaries. While the presence of both of these issues raises serious concerns in practical FL systems, no known schemes or combinations of schemes effectively address them at the same time. We propose Sageflow, staleness-aware grouping with entropy-based filtering and loss-weighted averaging, to handle both stragglers and adversaries simultaneously. Model grouping and weighting according to staleness (arrival delay) provides robustness against stragglers, while entropy-based filtering and loss-weighted averaging, working in a highly complementary fashion at each grouping stage, counter a wide range of adversary attacks. A theoretical bound is established to provide key insights into the convergence behavior of Sageflow. Extensive experimental results show that Sageflow outperforms various existing methods aiming to handle stragglers/adversaries.



## Requirements

This code was tested on the following environments:

* Ubuntu 18.04
* Python 3.7.13
* PyTorch 1.12.0
* CUDA 11.6

You can install all necessary packages from requirements.txt

```
pip install -r requirements.txt
```

## Experiments

* There are two experimental settings: normal setting (```cd``` to ```Sageflow_code```) and backdoor attack setting (```cd``` to ```Sageflow_backdoor_code```)
* Under each setting, experiments are run on 3 image classification datasets: MNIST, FMNIST and CIFAR-10 (these will be automatically downloaded from torchvision package). 

### How to Run

* ```cd``` to ```scripts/```
* Find a scenario to simulate and change parameters in the bash file to the values you want. 
* All parameters required for the experiment are described in ```options.py```. Please see the python file for a detailed description of the parameters.

```bash

# Scenario with only adversaries under normal setting

bash Sageflow_only_ad.sh

# Scenario with only stragglers and with both stragglers and adversaries under normal setting

bash Sageflow_both.sh

# Scenario with only adversaries under backdoor attack setting

bash Sageflow_backdoor_only_ad.sh

# Scenario with only stragglers and with both stragglers and adversaries under backdoor attack setting

bash Sageflow_backdoor_both.sh

```


## Citation

To cite Sageflow in your papers, please use the following bibtex entry.

```
@article{park2021sageflow,
  title={Sageflow: Robust federated learning against both stragglers and adversaries},
  author={Park, Jungwuk and Han, Dong-Jun and Choi, Minseok and Moon, Jaekyun},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={840--851},
  year={2021}
}
```

## Acknowledgement

Our code is built upon the implementations at https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/README.md
