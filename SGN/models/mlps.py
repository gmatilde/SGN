import os
import numpy as np
import random
import time
import argparse
import pickle

import theano
import theano.tensor as T

os.environ["THEANO_FLAGS"] = "floatX=float32"

class MLP(object):
    '''
    This class enables the realization of MLPs networks with variable architecture
    for classification.

    Attributes:

    hyper_par : a dictionary containg the hyperparameters in order to create the MLP network.
    '''
    def __init__(self, X, y_hat, input_size, output_size, hyper_par=None, mode='classification'):

        self.input_size = input_size
        self.output_size = output_size
        self.mode = mode

        if hyper_par is None:
            hyper_par = {}

        self.hyper_par = self.__check_hyper(hyper_par)
        self.units = self.__concat()
        self.N_par = self.__compute_Npar()
        self.X = X
        self.y_hat = y_hat
        theta0 = self.__init_par()
        self.theta = theano.shared(theta0.astype(theano.config.floatX), 'theta')
        self.W, self.b = self.__layers_par()
        self.out_sym, self.y_sym, self.pred_sym = self.__forward()
        self.forward = theano.function([self.X], [self.out_sym, self.y_sym, self.pred_sym])

    def __layers_par(self):

        idx = 0
        W = []
        b = []

        for ii in range(len(self.units)-1):

            W.append(self.theta[idx:idx+self.units[ii]*self.units[ii+1]].reshape((self.units[ii], self.units[ii+1])))
            idx += self.units[ii]*self.units[ii+1]
            b.append(self.theta[idx:idx+self.units[ii+1]])
            idx += self.units[ii+1]

        return W, b

    def __init_par(self):
        '''
        initialization of parameters vector theta
        '''
        theta0 = np.asarray([])
        np.random.seed(10)

        for ii in range(len(self.units)-1):

            W = 1*np.random.randn(self.units[ii]*self.units[ii+1])
            b = np.zeros(self.units[ii+1])
            theta0 = np.concatenate((theta0, W,b), axis=0)

        return theta0

    def __concat(self):
        '''
        miscellaneous function for concatenation of the dimensions.
        '''
        units = [self.input_size]

        for hidden_units in self.hyper_par['hs']:

            units.append(hidden_units)

        units.append(self.output_size)

        return units

    def __compute_Npar(self):
        '''
        compute number of parameters
        '''
        N_par = 0

        for ii in range(len(self.units)-1):

            N_par += self.units[ii+1]*(self.units[ii]+1)

        return N_par

    def __check_hyper(self, hyper_par):
        '''
        check the validity of the hyperparameters
        '''
        config = {
            'hs' : [32]
        }

        config.update(hyper_par)

        try:
            for x in config['hs']:
                if not(isinstance(x,int)) or x<=0:

                    raise Exception('The number of hidden units per hidden layers have to be positive integers.')

        except:

            raise ValueError('The definition of the hidden units per layer must be an iterable with integers, e.g. [64, 64] for a two layer MLP with 64 units in each layer')

        return(config)

    def __forward(self):
        '''
        feedforward propagation of network's input
        '''
        a = T.nnet.sigmoid(T.dot(self.X, self.W[0])+self.b[0])

        for ii in range(1, len(self.W)-1):

            a = T.nnet.sigmoid(T.dot(a, self.W[ii])+self.b[ii])

        if self.mode == 'classification':

            output = T.dot(a, self.W[-1])+self.b[-1]
            y = T.nnet.softmax(output)
            pred = T.argmax(y, axis=1)

        else:

            output = T.dot(a, self.W[-1])+self.b[-1]
            y = output
            pred = output

        return output, y, pred

if __name__ == '__main__':

    X = T.matrix(name='X', dtype=theano.config.floatX)
    y_hat = T.ivector('y_hat')

    net = MLP(X, y_hat, 10, 1, hyper_par = {'hs':[20, 20, 20]}, mode='regression')

    import pdb; pdb.set_trace()
