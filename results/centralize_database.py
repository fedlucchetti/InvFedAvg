import numpy as np
import matplotlib.pyplot as plt
import sys,json,os, glob
from os.path import join
from fnmatch import fnmatch



algorithm = "fedavg"


dir = algorithm


for subpath, subdirs, files in os.walk(join(".")):
    for filename in files:
        if fnmatch(filename, "*json"):
            filepath = join(subpath,filename)
            print(filepath)
