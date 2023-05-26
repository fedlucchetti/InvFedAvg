# Fedlearning


## Docker

Build Docker
```
./build_docker
```

Enter Docker Env (replace GPUID [0,1,...])
```
./run_docker $GPUID
```

## Create NON-IID data with different EMD values

```
python dirichlet_emd.py --nclients 5 --dataset cifar10 --alpha 1
```

## Train Autoencoder
```
python autoencoder.py --dataset cifar10 --bs 256 --rounds 40
```
