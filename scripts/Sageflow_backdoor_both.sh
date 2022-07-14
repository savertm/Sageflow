#!/bin/bash
cd ../Sageflow_backdoor_code/backdoor_FL

# For mnist, fmnist, epoch : 300
# For Cifar10, epoch : 2000

python Sageflow.py --update_rule Sageflow --epoch 300 --backdoor_scale 10 --dataset mnist --gpu_number 2 --eth 2 --delta 5 --lam 0.5 --ignore_straggler False --staleness 2





