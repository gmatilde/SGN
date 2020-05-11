import time
import json
import random
import pickle
import os
import time
import warnings
import argparse
import numpy as np

import theano
import theano.tensor as T

warnings.filterwarnings("ignore")# this is needed to deactivate a warning coming from numpy version

from SGN.optimizers.solvers import SGN, SGD
from SGN.models.vgg_nets import ConvNet
from auxiliaries import normalize_meanstd, str2bool, check_dataset, check_positive, check_rho_schedule, int_positive, check_solver, check_scheduler, check_gpu_cpu

os.environ["THEANO_FLAGS"] = "force_device=True, devide=cuda, floatX=float32, exception_verbosity=high"

################################################################################
parser = argparse.ArgumentParser(description='This script allows to train a \
convolutional neural network with SGN or SGD.')
#general settings
parser.add_argument('--path', type=str, help='path where folder for results will be \
saved', default='./')
parser.add_argument('--f', type=str, help='name of folder where results will be \
saved', default='results')
parser.add_argument('--solver', type=check_solver, help='solver', default='SGN')
parser.add_argument('--epochs', type=int_positive, help='number of epochs', default=1)
parser.add_argument('--batch_size', type=int_positive, help='batch size', default=1000)
parser.add_argument('--verbose', type=int_positive, help='printing level: there are four level of verbosity. \
With 0 no info will be printed, with 1 only general info about loss and accuracy, with 2 also info \
on the conjugate gradient iterations and gradient norm, with >=3 also info in \
the line search.', default=0)
parser.add_argument('--seed', type=int_positive, help='random seed', default=12)
parser.add_argument('--dataset', type=check_dataset, help='dataset to use', default='cifar10')
#hyper-parameters for SGN
parser.add_argument('--rho', type=check_positive, help='rho parameter for \
damping the Gauss-Newton metrix', default=10**-3)
parser.add_argument('--beta', type=check_positive, help='momentum parameter for SGN', default=0.0)
parser.add_argument('--cg_prec', type=str2bool, help='activate the preconditioner \
for the conjugate gradient method', default=False)
parser.add_argument('--LS', type=str2bool, help='activate the line search method', default=True)
parser.add_argument('--TR', type=str2bool, help='activate the trust region method', default=True)
parser.add_argument('--rho_schedule', type=check_rho_schedule, help='if trust region method is not active, \
the following decaying schedule for the rho damping parameter will be adopted', default='const')
parser.add_argument('--rho_min', type=check_positive, help='minimum value for \
rho parameter for damping the Gauss-Newton metrix', default=10**-6)
parser.add_argument('--cg_max_iter', type=int_positive, help='maximum number of \
conjugate gradient iterations', default=10)
parser.add_argument('--cg_min_iter', type=int_positive, help='minimum number of \
conjugate gradient iterations', default=10)
parser.add_argument('--K_cg', type=int_positive, help='after K_cg parameter updates (SGN iterations) \
the number of conjugate gradient iterations is increased from cg_min_iter to cg_max_iter', default=10)
parser.add_argument('--alpha_exp', type=check_positive, help='exponent for the \
preconditioner', default=1)
#hyper-parameters for SGD
parser.add_argument('--lr', type=check_positive, help='learning rate for SGD', default=0.001)
parser.add_argument('--mom', type=check_positive, help='momentum parameter for SGD', default=0.0)
parser.add_argument('--scheduler', type=check_scheduler, help='scheduler for the learning \
rate decay', default='const')
parser.add_argument('--lr_final', type=check_positive, help='final learning rate \
for SGD', default=0.001)
parser.add_argument('--K', type=int_positive, help='final iteration for decaying', default=1000)
parser.add_argument('--r', type=check_positive, help='rate of learning rate \
decay for step decying scheduler', default=0.1)
parser.add_argument('--step', type=int_positive, help='step width for step \
decaying scheduler', default=10)
#name of json file with the architecture structure
parser.add_argument('--architecture', type=str, help='json file containing the \
architecture info', default='architecture_2')

args = parser.parse_args()
################################################################################
np.random.seed(args.seed)
random.seed(args.seed)
print('random seed {}'.format(args.seed))
################################################################################
check_gpu_cpu()
################################################################################
if not os.path.exists(os.path.join(args.path, args.f)):
    os.makedirs(os.path.join(args.path, args.f))
elif os.path.exists(os.path.join(args.path, args.f, 'results.json')):
    raise AssertionError('name of the folder {} is not valid as it already exists in {}.'.format(args.f, args.path))
################################################################################
print('Extract the data...\n')


if args.dataset == 'cifar10':
    from keras.datasets import cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    train_images = np.moveaxis(np.float32(train_images), 3, 1)
    test_images = np.moveaxis(np.float32(test_images), 3, 1)
    train_labels = np.squeeze(train_labels)
    test_labels = np.squeeze(test_labels)
elif args.dataset == 'fashion':
    from keras.datasets import fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = np.expand_dims(train_images, axis=1)
    test_images = np.expand_dims(test_images, axis=1)
else:
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = np.expand_dims(train_images, axis=1)
    test_images = np.expand_dims(test_images, axis=1)

n_classes = 10

#preprocess data function (standardization)
train_images = normalize_meanstd(train_images, axis=(2,3))
test_images = normalize_meanstd(test_images, axis=(2,3))

print('\n{} training samples, {} test samples'.format(train_images.shape[0], test_images.shape[0]))

n_channels = train_images.shape[1]
height = train_images.shape[2]
width = train_images.shape[3]

print('\nnumber of channels {}, height {}, width {}, number of classes {}\n'.format(n_channels, height, width, n_classes))
################################################################################
print('Build the Graph...\n')

X = T.tensor4(name='X', dtype=theano.config.floatX)
y_hat = T.ivector('y_hat') # Theano integer vector

#import the architecture structure
try:
    with open(os.path.join('architectures','{}.json'.format(args.architecture)), 'r') as f:
        architecture = json.load(f)

        conv_layers = architecture['conv_layers']
        mlp_layers = architecture['mlp_layers']

except:
    print('no file {} found, using the default architecture.'.format(args.architecture))

    conv_layers = {'layer0':{'type':'CONV', 'num_filters':6, 'filter_size':5},
                   'layer1':{'type':'ACT', 'activation':'relu'},
                   'layer2':{'type':'POOL'},
                   'layer3':{'type':'CONV', 'num_filters':16, 'filter_size':5},
                   'layer4':{'type':'ACT', 'activation':'relu'},
                   'layer5':{'type':'POOL'}
                  }
    mlp_layers  = {'layer0':{'type':'FC', 'output_size':120, 'activation':'relu'},
                   'layer1':{'type':'FC', 'output_size':84, 'activation':'relu'},
                  }

net = ConvNet(X, y_hat, (n_channels,height,width), n_classes, conv_layers = conv_layers, mlp_layers = mlp_layers)

print('number of paramters {}\n'.format(net.N_par))

#create the loss expression
categorical_crossentropy = -T.mean(T.log(net.y_sym)[T.arange(net.y_sym.shape[0]), y_hat])

if args.solver == 'SGN':

    hyper_config = {
    'SOLVER': args.solver,
    'MAX_CG_ITERS': args.cg_max_iter,
    'MIN_CG_ITERS': args.cg_min_iter,
    'K_CG': args.K_cg,
    'TR': args.TR,
    'LS': args.LS,
    'CG_TOL': 10**-6,
    'PRINT_LEVEL': args.verbose,
    'MAX_LS_ITERS': 10,
    'LS': True,
    'CG_PREC': args.cg_prec,
    'DAMP_RHO': args.rho,
    'RHO_LS': 0.5,
    'C_LS': 10**-4,
    'ALPHA': 1,
    'BETA': args.beta,
    'DAMP_RHO_MIN': args.rho_min,
    'BATCH_SIZE': args.batch_size,
    'ALPHA_EXP': args.alpha_exp
    }

    if not(args.TR):
        hyper_config['RHO_DECAY'] = args.rho_schedule
        hyper_config['K'] = args.K

    optimizer = SGN(X, y_hat, net, categorical_crossentropy, hyper_par=hyper_config)

else:

    hyper_config = {
    'SOLVER': args.solver,
    'MOMENTUM': args.mom,
    'LR0': args.lr, #initial learning rate
    'SCHEDULER': args.scheduler, #scheduling of the learning rate (costant, linear decay, step decay, cosine decay),
    'LR_K': args.lr_final,
    'K': args.K,
    'EPOCHS': args.epochs,
    'BATCH_SIZE': args.batch_size
    }

    if args.scheduler == 'step':
        hyper_config['STEP'] = args.step
        hyper_config['REDUCTION'] = args.r

    optimizer = SGD(X, y_hat, net, categorical_crossentropy, hyper_par=hyper_config)

#save the configurations
res_folder = os.path.join(args.path, args.f)

hyper_par = {'config': hyper_config, 'mlp_layers': mlp_layers, 'conv_layers': conv_layers}
#save the architecture and config parameters
with open(os.path.join(res_folder, 'config.json'), 'w') as f:
    json.dump(hyper_par, f)

################################################################################
print('Start Training...\n')

N_data = train_images.shape[0]
N_test = test_images.shape[0]
idx_data = np.array(range(N_data)).tolist()

nepochs = args.epochs
batch_size = args.batch_size

epoch = 0
tot_iter = 0

res_acc = []
res_testacc = []
res_loss = []

start_time = time.time()

while (epoch<nepochs):

    epoch += 1
    iteration = 0

    random.shuffle(idx_data)
    idx_tmp = idx_data

    loss_per_epoch = 0.0
    correct = 0.0

    while(len(idx_tmp)>0):

        iteration += 1
        tot_iter += 1
        #select the mini-batch of images
        M_value = min(batch_size, len(idx_tmp))
        images_batch = train_images[idx_tmp[0:M_value]]
        labels_batch = train_labels[idx_tmp[0:M_value]]
        idx_tmp = idx_tmp[M_value:]

        loss, grad, pred = optimizer.step(images_batch, labels_batch)

        loss_per_epoch += loss.item()
        correct += (pred == labels_batch).sum().item()/len(labels_batch)

        if tot_iter % 5 == 0: #print every 5 iterations
            if args.verbose>=1:
                print('[%d, %5d] loss: %.3f, train acc: %.2f %%' %
                      (epoch, iteration, loss_per_epoch/iteration, (correct/iteration)*100))

        if tot_iter % 50 == 0: #compute the test accuracy every 50 iterations
            _, _, pred = net.forward(test_images)
            test_acc = (pred == test_labels).sum()/N_test
            if args.verbose>=1:
                print('\n====> test accuracy {:2} %'.format(test_acc*100))

    #saving results per epoch
    res_loss.append(loss_per_epoch/iteration)
    res_acc.append(correct/iteration)
    _, _, pred = net.forward(test_images)
    test_acc = (pred == test_labels).sum()/N_test
    res_testacc.append(test_acc)

res = {'avg_loss':res_loss, 'acc': res_acc, 'test_acc': res_testacc, 'tot_time': time.time()-start_time, 'tot_iterations': tot_iter, 'tot_epochs':epoch}
#sgn
if args.solver == 'SGN':
    res['rho'] = optimizer.damp_rho
    res['alpha'] = optimizer.alpha

#save the results
with open(os.path.join(res_folder, 'results.json'), 'w') as f:
    json.dump(res, f)
