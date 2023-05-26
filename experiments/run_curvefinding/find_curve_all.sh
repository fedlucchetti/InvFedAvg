#!/bin/bash
# cd ../


for bs in 128 256
do
  for le in 1 2 5 10
  do
      python model_interaction.py --n_local_runs 20 --local_batchsize $bs --local_epochs $le --cf_nruns 0
  done
done
