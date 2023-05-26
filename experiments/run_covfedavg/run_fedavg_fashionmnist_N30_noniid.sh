cd ../

for i in 0 1 2 3 4
do
  python3  main.py --local_epochs 1 --nclients 30 --batchsize 128 \
                  --rounds 100 --dataset fashion_mnist --emd 2.3 \
                  --emd_l 0.1  --algorithm fedavg --nu 0.0001  \
                  --runID $i
done

for i in 100 101 102 103 104
do
  python3  main.py --local_epochs 1 --nclients 30 --batchsize 128 \
                  --rounds 100 --dataset fashion_mnist --emd 3.2 \
                  --emd_l 0.1  --algorithm fedavg --nu 0.0001  \
                  --runID $i
done
