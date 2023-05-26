from fedlearning import autoencoder
import sys, json, os

try:
    for idx,arg in enumerate(sys.argv):
        if arg=="--dataset" or arg=="--ds":
            dataset         = sys.argv[idx+1]
        if arg=="--batchsize" or arg=="--bs":
            batch_size         = int(sys.argv[idx+1])
        if arg=="--rounds" or arg=="--r":
            rounds         = int(sys.argv[idx+1])
except Exception as e:
    print(e)
    sys.exit()

ae = autoencoder.Autoencoder(dataset=dataset,batch_size=batch_size,rounds=rounds)
ae.run()
