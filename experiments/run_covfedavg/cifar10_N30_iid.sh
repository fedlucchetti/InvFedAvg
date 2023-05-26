#!/bin/bash
# cd ../


for i in 0 1 2 3 4
do
  python3  main.py --local_epochs 2 --nclients 30 --batchsize 128 \
                  --rounds 100 --dataset cifar10 --emd 0.3  \
                  --emd_l 0.0  --algorithm covfedavg --nu 0.0001  \
                  --runID $i
done
