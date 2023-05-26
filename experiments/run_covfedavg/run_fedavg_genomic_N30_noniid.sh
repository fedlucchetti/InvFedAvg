#!/bin/bash
cd ../

for i in 0 1 2 3 4
do
 python3  main.py --local_epochs 2 --nclients 30 --batchsize 128 \
                 --rounds 100 --dataset genomic --emd 3.7 \
                 --emd_l 1.5  --algorithm fedavg --nu 0.0001  \
                 --runID $i
done

for i in 100 101 102 103 104
do
  python3  main.py --local_epochs 2 --nclients 30 --batchsize 128 \
                  --rounds 100 --dataset genomic --emd 4.8 \
                  --emd_l 1.5  --algorithm fedavg --nu 0.0001  \
                  --runID $i
done
