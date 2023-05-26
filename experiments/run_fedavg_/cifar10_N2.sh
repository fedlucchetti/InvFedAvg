#!/bin/bash
# cd ../



python3  main.py --local_epochs 3 --nclients 2 --local_batchsize 128 \
                --rounds 100 --dataset cifar10 --alpha 1  \
                --ni 0.42  --algorithm fedavg --flag_common_weight_init 0

# python3  main.py --local_epochs 3 --nclients 2 --local_batchsize 128 \
#                 --rounds 100 --dataset cifar10 --alpha 1  \
#                 --ni 0.5  --algorithm fedavg 

# python3  main.py --local_epochs 3 --nclients 2 --local_batchsize 128 \
#                 --rounds 100 --dataset cifar10 --alpha 1  \
#                 --ni 0.55  --algorithm fedavg 

# python3  main.py --local_epochs 3 --nclients 2 --local_batchsize 128 \
#                 --rounds 100 --dataset cifar10 --alpha 0.1  \
#                 --ni 0.42  --algorithm fedavg 

# python3  main.py --local_epochs 3 --nclients 2 --local_batchsize 128 \
#                 --rounds 100 --dataset cifar10 --alpha 10  \
#                 --ni 0.42  --algorithm fedavg 