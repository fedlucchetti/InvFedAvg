#!/bin/bash
cd ../

for i in 200 201 202 203 204
do
  python3  main.py --local_epochs 2 --nclients 30 --batchsize 64 \
                  --rounds 100 --dataset genomic --emd 0.0 \
                  --emd_l 0.0  --algorithm fedavg --nu 0.0001  \
                  --runID $i
done
