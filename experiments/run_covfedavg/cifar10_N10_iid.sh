#!/bin/bash
# cd ../


for i in 0
do
  python3  main.py --local_epochs 1 --nclients 10 --batchsize 128 \
                  --rounds 100 --dataset cifar10 --emd 0.1  \
                  --emd_l 0.0  --algorithm covfedavg --nu 0.0001  \
                  --runID $i
done
