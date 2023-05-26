#!/bin/bash
# cd ../


for i in 1 
do
    python3  main.py --local_epochs 5 --nclients 30 --local_batchsize 32 \
                    --rounds 200 --dataset cifar10 --alpha 0.3 --lr 0.0001  \
                    --ni 0.75  --algorithm fedavg --flag_common_weight_init 0

    python3  main.py --local_epochs 5 --nclients 30 --local_batchsize 32 \
                    --rounds 200 --dataset cifar10 --alpha 0.3 --lr 0.0001  \
                    --ni 1.2  --algorithm fedavg  --flag_common_weight_init 0

    python3  main.py --local_epochs 5 --nclients 30 --local_batchsize 32 \
                    --rounds 200 --dataset cifar10 --alpha 0.3 --lr 0.0001  \
                    --ni 1.71  --algorithm fedavg  --flag_common_weight_init 0

    python3  main.py --local_epochs 5 --nclients 30 --local_batchsize 32 \
                    --rounds 200 --dataset cifar10 --alpha 1 --lr 0.0001  \
                    --ni 0.69  --algorithm fedavg --flag_common_weight_init 0

    python3  main.py --local_epochs 5 --nclients 30 --local_batchsize 32 \
                    --rounds 200 --dataset cifar10 --alpha 10 --lr 0.0001  \
                    --ni 0.69  --algorithm fedavg --flag_common_weight_init 0
done