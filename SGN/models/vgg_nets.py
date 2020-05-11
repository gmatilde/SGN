
import os
import random
import numpy as np

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

os.environ["THEANO_FLAGS"] = "floatX=float32, exception_verbosity=high"

################################################################################
'''
modular implementation based on ConvOp: this is the main building block for
implementing a convolutional layer in Theano.
It is used by theano.tensor.signal.conv2d, which is used here and which takes as
inputs

1) a 4D tensor corresponding to a mini-batch of input images. The shape of the tensor is
[mini-batch size, number of input feature maps, image height, image width]

2) a 4D tensor corresponding to the weight matrix W. The shape of the tensor is
[number of feature maps at layer m, number of feature maps at layer m-1, filer height,
filter width]
'''
################################################################################
'''
an example of possible structures:

conv_layers = {
          'layer0': {'type':'CONV', 'num_filters':64, 'filter_size': 9, 'padding': 0, 'stride':1},
          'layer1': {'type':'CONV', 'num_filters':64, 'filter_size': 9, 'padding': 0, 'stride':1},
          'layer2': {'type':'CONV', 'num_filters':64, 'filter_size': 9, 'padding': 0, 'stride':1},
          'layer3': {'type':'POOL', 'pool_size':2},
          'layer4': {'type':'ACT', 'activation':'relu'}
           }

mlp_layers = {
          'layer0': {'type':'FC', 'output_size':32, 'activation':'relu'},
          'layer1': {'type':'FC', 'output_size':64, 'activation':'relu'},
          }

'''
################################################################################

class ConvNet(object):

    def __init__(self, X, y_hat, input_size, output_size, conv_layers=None, mlp_layers=None):

        if len(input_size)!=3:
            raise Exception('The input size should be [number of channels, image height, image width]')

        self.input_size = input_size
        self.output_size = output_size

        if conv_layers is None:

            self.conv_net = {
            'layer0': {'type':'CONV', 'num_filters':64, 'filter_size': 9, 'padding': 0, 'stride':1},
            'layer1': {'type':'CONV', 'num_filters':64, 'filter_size': 9, 'padding': 0, 'stride':1},
            'layer2': {'type':'CONV', 'num_filters':64, 'filter_size': 9, 'padding': 0, 'stride':1},
            'layer3': {'type':'ACT', 'activation': 'relu'},
            'layer4': {'type':'POOL', 'pool_size':2},
            }

        else:

            self.conv_net = self.__check_structure(conv_layers, 'conv')

        if mlp_layers is None:

            self.mlp_net = {
            'layer0': {'type': 'FC', 'output_size': 10, 'activation': 'relu'},
            }

        else:

            self.mlp_net = self.__check_structure(mlp_layers, 'mlp')

        self.N_par = self.__compute_Npar()
        self.X = X
        theta0 = self.__init_par()
        self.theta = theano.shared(theta0.astype(theano.config.floatX), 'theta')
        self.W, self.b = self.__split_par()
        self.out_sym, self.y_sym, self.pred_sym = self.__forward()
        self.forward = theano.function([self.X], [self.out_sym, self.y_sym, self.pred_sym])

    def __check_structure(self, net_structure, type_net):

        if type_net == 'conv':

            layer_types = ['CONV', 'POOL', 'ACT']

        elif type_net == 'mlp':

            layer_types = ['FC']

        available_activations = ['relu', 'sigmoid']

        N_layers = len(net_structure)

        for ii in range(N_layers):

            conv_ = {
            'num_filters': 10,
            'filter_size': 3,
            'padding': 0,
            'stride': 1
            }

            fc_ = {
            'output_size': 10,
            'activation': 'relu'
            }

            pool_ = {
            'pool_size': 2,
            'stride': 2
            }

            act_ = {
            'activation': 'relu'
            }

            key_ii = 'layer'+str(ii)

            if key_ii not in net_structure:

                raise Exception('wrong keys in {}'.format(net_structure))

            if 'type' not in net_structure[key_ii]:

                raise Exception('the type of layer must be explicitly specified with the key \'type\'')

            if net_structure[key_ii]['type'] not in layer_types:

                raise ValueError('{} is not available for {} net type. Please choose among the following layer types {}'.format(net_structure[key_ii], type_net, layer_types))

            elif net_structure[key_ii]['type']=='CONV':

                conv_.update(net_structure[key_ii])
                net_structure[key_ii] = conv_

                #num_filters--> integer and positive
                if not(isinstance(net_structure[key_ii]['num_filters'], int)) or net_structure[key_ii]['num_filters']<=0:

                    raise Exception('The number of filters has to be a positive integer.')

                #filter size--> integer and positive
                if not(isinstance(net_structure[key_ii]['filter_size'], int)) or net_structure[key_ii]['filter_size']<=0:

                    raise Exception('The size of filters has to be a positive integer.')

                #padding--> integer and positive
                if not(isinstance(net_structure[key_ii]['padding'], int)) or net_structure[key_ii]['padding']<0:

                    raise Exception('The padding has to be a non-negative integer.')

                #strides--> integer and positive
                if not(isinstance(net_structure[key_ii]['stride'], int)) or net_structure[key_ii]['stride']<=0:

                    raise Exception('The stride has to be a positive integer.')


            elif net_structure[key_ii]['type']=='POOL':

                pool_.update(net_structure[key_ii])
                net_structure[key_ii] = pool_

                #pool_size--> integer and positive
                if not(isinstance(net_structure[key_ii]['pool_size'], int)) or net_structure[key_ii]['pool_size']<=0:

                    raise Exception('The pool shape has to be a positive integer.')

                #stride--> integer and positive
                if not(isinstance(net_structure[key_ii]['stride'], int)) or net_structure[key_ii]['stride']<=0:

                    raise Exception('The stride has to be a positive integer.')

            elif net_structure[key_ii]['type']=='ACT':

                act_.update(net_structure[key_ii])
                net_structure[key_ii] = act_

                if not net_structure[key_ii]['activation'] in available_activations:

                    raise Exception('{} is not available as activation function. Please choose among {}'.format(net_structure[key_ii]['activation'], available_activations))

            else:

                fc_.update(net_structure[key_ii])
                net_structure[key_ii] = fc_

                #output_size--> integer and positive
                if not(isinstance(net_structure[key_ii]['output_size'], int)) or net_structure[key_ii]['output_size']<=0:

                    raise Exception('The output size has to be a positive integer.')

                #activation--> in list
                if net_structure[key_ii]['activation'] not in available_activations:

                    raise Exception('{} is not available as activation function. Please choose among {}'.format(net_structure[key_ii]['activation'], available_activations))

        return net_structure

    def __compute_Npar(self):

        n_channels = self.input_size[0]
        height = self.input_size[1]
        width = self.input_size[2]

        counter = 0

        for ii in range(len(self.conv_net)):

            key_ii = 'layer'+str(ii)

            if self.conv_net[key_ii]['type']=='CONV':

                counter += self.conv_net[key_ii]['num_filters']*n_channels*(self.conv_net[key_ii]['filter_size']**2)
                counter += self.conv_net[key_ii]['num_filters']
                n_channels = self.conv_net[key_ii]['num_filters']
                width = int((width - self.conv_net[key_ii]['filter_size'] + 2*self.conv_net[key_ii]['padding'])/self.conv_net[key_ii]['stride'] + 1)
                height = int((height - self.conv_net[key_ii]['filter_size'] + 2*self.conv_net[key_ii]['padding'])/self.conv_net[key_ii]['stride'] + 1)

            if self.conv_net[key_ii]['type'] == 'POOL':

                width = int(width/self.conv_net[key_ii]['pool_size'])
                height = int(height/self.conv_net[key_ii]['pool_size'])

        input_size = width*height*n_channels

        for ii in range(len(self.mlp_net)):

            key_ii = 'layer'+str(ii)
            counter += input_size*self.mlp_net[key_ii]['output_size']
            counter += self.mlp_net[key_ii]['output_size']
            input_size = self.mlp_net[key_ii]['output_size']

        counter += input_size*self.output_size
        counter += self.output_size

        return int(counter)

    def __init_par(self):

        rng = np.random.RandomState(1)

        #xavier initialization of the weights
        W = []
        b = []
        theta0 = np.asarray([])
        n_channels = self.input_size[0]
        width = self.input_size[1]
        height = self.input_size[2]

        #loop over layers in the convolutional part of the net
        for ii in range(len(self.conv_net)):

            key_ii = 'layer'+str(ii)

            if self.conv_net[key_ii]['type'] == 'CONV':

                w_bound = self.conv_net[key_ii]['num_filters']*n_channels*self.conv_net[key_ii]['filter_size']*self.conv_net[key_ii]['filter_size']
                w_shp = (self.conv_net[key_ii]['num_filters'], n_channels, self.conv_net[key_ii]['filter_size'], self.conv_net[key_ii]['filter_size'])
                W_ii = np.asarray(rng.uniform(
                       low = -np.sqrt(6./w_bound),
                       high = +np.sqrt(6./w_bound),
                       size = w_shp
                ))
                W_ii = W_ii.reshape(w_bound)
                b_ii = np.zeros((self.conv_net[key_ii]['num_filters'], ))
                theta0 = np.concatenate((theta0, W_ii, b_ii), axis=0)
                n_channels = self.conv_net[key_ii]['num_filters']
                width = int((width - self.conv_net[key_ii]['filter_size'] + 2*self.conv_net[key_ii]['padding'])/self.conv_net[key_ii]['stride'] +1)
                height = int((height - self.conv_net[key_ii]['filter_size'] + 2*self.conv_net[key_ii]['padding'])/self.conv_net[key_ii]['stride'] +1)

            if self.conv_net[key_ii]['type'] == 'POOL':

                width = int(width/self.conv_net[key_ii]['pool_size'])
                height = int(height/self.conv_net[key_ii]['pool_size'])

        input_size = int(width*height*n_channels)

        #loop over layers in the feedforward part of the network
        for ii in range(len(self.mlp_net)):

            key_ii = 'layer'+str(ii)
            w_bound = input_size*self.mlp_net[key_ii]['output_size']
            w_shp = (input_size, self.mlp_net[key_ii]['output_size'])
            W_ii = np.asarray(rng.uniform(
            low = -np.sqrt(6./(input_size+self.output_size)),
            high = +np.sqrt(6./(input_size+self.output_size)),
            size = w_shp
            ))
            W_ii = W_ii.reshape(w_bound)
            b_ii = np.zeros((self.mlp_net[key_ii]['output_size'],))
            theta0 = np.concatenate((theta0, W_ii, b_ii), axis=0)
            input_size = int(self.mlp_net[key_ii]['output_size'])

        #output layer
        w_bound = input_size*self.output_size
        w_shp = (input_size, self.output_size)
        W_ii = np.asarray(rng.uniform(
        low = -np.sqrt(6./(input_size+self.output_size)),
        high = +np.sqrt(6./(input_size+self.output_size)),
        size = w_shp
        ))
        W_ii = W_ii.reshape(w_bound)
        b_ii = np.zeros((self.output_size,))
        theta0 = np.concatenate((theta0, W_ii, b_ii), axis=0)

        return theta0

    def __split_par(self):

        n_channels = self.input_size[0]
        width = self.input_size[1]
        height = self.input_size[2]
        idx = 0
        W = []
        b = []

        for ii in range(len(self.conv_net)):

            key_ii = 'layer'+str(ii)

            if self.conv_net[key_ii]['type'] == 'CONV':

                W.append(self.theta[idx:idx+n_channels*self.conv_net[key_ii]['num_filters']*self.conv_net[key_ii]['filter_size']**2].reshape((self.conv_net[key_ii]['num_filters'], n_channels,self.conv_net[key_ii]['filter_size'], self.conv_net[key_ii]['filter_size'])))
                idx += n_channels*self.conv_net[key_ii]['num_filters']*self.conv_net[key_ii]['filter_size']**2
                b.append(self.theta[idx:idx+self.conv_net[key_ii]['num_filters']])
                idx += self.conv_net[key_ii]['num_filters']
                #update number of channels, width and height
                n_channels = self.conv_net[key_ii]['num_filters']
                width = int((width - self.conv_net[key_ii]['filter_size'] +2*self.conv_net[key_ii]['padding'])/self.conv_net[key_ii]['stride'] +1)
                height = int((height - self.conv_net[key_ii]['filter_size'] +2*self.conv_net[key_ii]['padding'])/self.conv_net[key_ii]['stride'] +1)

            if self.conv_net[key_ii]['type'] == 'POOL':

                width = int(width/self.conv_net[key_ii]['pool_size'])
                height = int(height/self.conv_net[key_ii]['pool_size'])

        input_size = int(width*height*n_channels)

        for ii in range(len(self.mlp_net)):

            key_ii = 'layer'+str(ii)
            W.append(self.theta[idx:idx+input_size*self.mlp_net[key_ii]['output_size']].reshape((input_size, self.mlp_net[key_ii]['output_size'])))
            idx += input_size*self.mlp_net[key_ii]['output_size']
            b.append(self.theta[idx:idx+self.mlp_net[key_ii]['output_size']])
            idx += self.mlp_net[key_ii]['output_size']
            input_size = self.mlp_net[key_ii]['output_size']

        #output layer
        W.append(self.theta[idx:idx+input_size*self.output_size].reshape((input_size, self.output_size)))
        idx += input_size*self.output_size
        b.append(self.theta[idx:idx+self.output_size])
        idx += self.output_size

        return W, b

    def __forward(self):

        count = 0
        input = self.X

        #first loop over the conv net
        for ii in range(len(self.conv_net)):

            key_ii = 'layer'+str(ii)

            if self.conv_net[key_ii]['type'] =='CONV':

                conv_out = conv2d(input, self.W[count], filter_flip=False, border_mode=self.conv_net[key_ii]['padding'])#for now only stride 1
                output = conv_out + self.b[count].dimshuffle('x', 0, 'x', 'x')
                count += 1

            elif self.conv_net[key_ii]['type'] =='ACT':

                if self.conv_net[key_ii]['activation'] == 'relu':

                    output = T.nnet.relu(output)

                elif self.conv_net[key_ii]['activation'] == 'sigmoid':

                    output = T.nnet.sigmoid(output)

            elif self.conv_net[key_ii]['type'] =='POOL':

                poolsize = (self.conv_net[key_ii]['pool_size'], self.conv_net[key_ii]['pool_size'])
                output = pool.pool_2d(output, poolsize, ignore_border=True)
            input = output

        #flatten before applying the fully connected layer
        output = output.flatten(2)

        #second loop over the mlp net
        for ii in range(len(self.mlp_net)):

            key_ii = 'layer'+str(ii)
            output = T.dot(output, self.W[count])+self.b[count]
            if self.mlp_net[key_ii]['activation'] == 'relu':

                output = T.nnet.relu(output)

            elif self.mlp_net[key_ii]['activation'] == 'sigmoid':

                output = T.nnet.sigmoid(output)

            count += 1

        #output of the last layer
        output = T.dot(output, self.W[-1])+self.b[-1]
        y = T.nnet.softmax(output)
        pred = T.argmax(y, axis=1)

        return output, y, pred

if __name__ == '__main__':

    conv_layers = {
              'layer0': {'type':'CONV', 'num_filters': 16, 'filter_size': 3, 'padding': 1, 'stride': 1},#for now only stride 1 supported TODO:add more options for stride
              'layer1': {'type':'CONV', 'num_filters': 16, 'filter_size': 3, 'padding': 1, 'stride': 1},
              'layer2': {'type':'CONV', 'num_filters': 16, 'filter_size': 3, 'padding': 1, 'stride': 1},
              'layer3': {'type':'CONV', 'num_filters': 16, 'filter_size': 3, 'padding': 1, 'stride': 1},
              'layer4': {'type':'POOL', 'pool_size':2},#for now only max pooling supported TODO: add different types of pooling, e.g. averaging. Also stride only equal to pool_size TODO: add different strides options 
              'layer5': {'type':'ACT', 'activation':'sigmoid'}
    }

    mlp_layers = {
              'layer0': {'type':'FC', 'output_size':32, 'activation':'relu'},
              'layer1': {'type':'FC', 'output_size':32, 'activation':'relu'},
              }

    input_size = [3, 10, 10]
    output_size = 2
    X = T.tensor4(name='X', dtype=theano.config.floatX)
    y_hat = T.ivector('y_hat')

    net = ConvNet(X, y_hat, input_size, output_size, conv_layers=conv_layers, mlp_layers=mlp_layers)

    print('number of parameters {}'.format(net.N_par))
