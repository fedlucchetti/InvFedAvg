#!/bin/bash
# cd ../


for i in 1 2 3 4
do
    python3  main.py --local_epochs 3 --nclients 10 --local_batchsize 128 \
                    --rounds 150 --dataset cifar10 --alpha 0.1 --lr 0.0001  \
                    --ni 0.5  --algorithm fedavg 

    python3  main.py --local_epochs 3 --nclients 10 --local_batchsize 128 \
                    --rounds 150 --dataset cifar10 --alpha 0.1 --lr 0.0001  \
                    --ni 0.75  --algorithm fedavg 

    python3  main.py --local_epochs 3 --nclients 10 --local_batchsize 128 \
                    --rounds 150 --dataset cifar10 --alpha 0.1 --lr 0.0001  \
                    --ni 1.0  --algorithm fedavg 

    python3  main.py --local_epochs 3 --nclients 10 --local_batchsize 128 \
                    --rounds 150 --dataset cifar10 --alpha 0.1 --lr 0.0001  \
                    --ni 1.37  --algorithm fedavg


    python3  main.py --local_epochs 3 --nclients 10 --local_batchsize 128 \
                    --rounds 150 --dataset cifar10 --alpha 1.0 --lr 0.0001  \
                    --ni 0.75  --algorithm fedavg 

    python3  main.py --local_epochs 3 --nclients 10 --local_batchsize 128 \
                    --rounds 150 --dataset cifar10 --alpha 10 --lr 0.0001  \
                    --ni 0.75  --algorithm fedavg 

done