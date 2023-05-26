#!/bin/bash

for i in 0 1 2 3 4
do
  python3  main.py --local_epochs 3 --nclients 30 --batchsize 128 \
                  --rounds 100 --dataset imdb --emd 2.7 \
                  --emd_l 0.1  --algorithm fedavg --nu 0.0001  \
                  --runID $i
done

for i in 100 101 102 103 104
do
  python3  main.py --local_epochs 3 --nclients 30 --batchsize 128 \
                  --rounds 100 --dataset imdb --emd 3.8 \
                  --emd_l 0.1  --algorithm fedavg --nu 0.0001  \
                  --runID $i
done
