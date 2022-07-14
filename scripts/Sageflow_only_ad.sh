#!/bin/bash
cd ../Sageflow_code

# For mnist and fmnist, epoch : 150 and lrdecay : 7
# For Cifar10, epoch : 1200 and lrdecay : 300


python Sageflow_synchronous_setup.py --epoch 150 --lrdecay 7 --update_rule Sageflow --data_poison False --model_poison True --dataset fmnist --frac 0.2 --attack_ratio 0.2 --gpu_number 7 --iid 2 --model_poison_scale 0.1 --eth 1 --delta 1 --seed 2022

