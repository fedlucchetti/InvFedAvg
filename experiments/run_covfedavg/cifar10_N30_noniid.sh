#!/bin/bash
#cd ../

for i in 3000 3001 3002 3003 3004
do
  python3  main.py --local_epochs 2 --nclients 30 --batchsize 128 \
                  --rounds 100 --dataset cifar10 --emd 2.3  \
                  --emd_l 0.1  --algorithm covfedavg --nu 0.0001  \
                  --runID $i
done

for i in 3100 3101 3102 3103 3104
do
  python3  main.py --local_epochs 2 --nclients 30 --batchsize 128 \
                  --rounds 100 --dataset cifar10 --emd 1.7  \
                  --emd_l 0.1  --algorithm covfedavg --nu 0.0001  \
                  --runID $i
done
