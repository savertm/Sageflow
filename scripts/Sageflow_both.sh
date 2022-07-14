#!/bin/bash
cd ../Sageflow_code

# For mnist and fmnist, epoch 150
# For cifar10, epoch: 1200

python Sageflow.py --epoch 150 --update_rule Sageflow --lrdecay 2000 --data_poison False --model_poison True --dataset mnist --frac 0.2 --attack_ratio 0.2 --gpu_number 4 --iid 2 --model_poison_scale 0.1 --eth 1 --delta 1  --lam 0.5 --seed 2021 --staleness 2
