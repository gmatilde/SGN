import numpy as np
import matplotlib.pyplot as plt
import random
import json
import argparse


parser = argparse.ArgumentParser(description='Sample points from a sinusoidal\
wave function corrupted by Gaussian noise.')

parser.add_argument('--noise', type=float, default=0.01,
                    help='noise level')
parser.add_argument('--freq', type=float, default=10,
                    help='frequency of sinusoidal wave function')
parser.add_argument('--N', type=int, default=100,
                    help='number of samples')

args = parser.parse_args()

X = np.random.uniform(-1,1,size=(args.N, 1))

Y = np.sin(args.freq*X)[:, 0] + np.random.normal(0, args.noise, size=(args.N,))

N_test = int(args.N*0.3)

data = {'X_train': X[N_test:, :].tolist(), 'y_train': Y[N_test:].tolist(), 'X_test': X[:N_test, :].tolist(), 'y_test': Y[:N_test].tolist()}

with open('sinusoidal_data_{}_{}.json'.format(args.freq, args.noise), 'w') as f:

    json.dump(data, f)

f.close()
