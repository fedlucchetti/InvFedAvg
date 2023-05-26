#!/bin/bash
# cd ../


# for emd in 2 30
# do
#   for le in 2 5 10 20
#   do
#     for mcs in 1024 2048 4096
#     do
#       python3  curve_finding.py  --batchsize 128  --nclients 2 \
#                                  --rounds 1 --dataset cifar10 --nu 0.001 \
#                                  --local_epochs $le --emd $emd \
#                                  --algorithm covfedavg  --mc_size $mcs
#     done
#   done
# done


python3  curve_finding.py  --batchsize 128  --nclients 2 \
                           --rounds 1 --dataset cifar10 --nu 0.001 \
                           --local_epochs 2 --emd 30 \
                           --algorithm covfedavg  --mc_size 1024
