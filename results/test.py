import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--example', nargs='?', const=1, type=int,default=10)
args = parser.parse_args()
print(args)
