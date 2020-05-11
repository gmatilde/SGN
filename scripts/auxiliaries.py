import numpy
import time
import argparse
import theano

def check_gpu_cpu():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000
    rng = numpy.random.RandomState(22)
    x = theano.shared(numpy.asarray(rng.rand(vlen), theano.config.floatX))
    f = theano.function([], theano.tensor.exp(x))
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Running a short test...")
    if numpy.any([isinstance(x.op, theano.tensor.Elemwise) and
                  ('Gpu' not in type(x.op).__name__)
                  for x in f.maker.fgraph.toposort()]):
        print('Using the cpu')
    else:
        print('Using the gpu')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_positive(v):
    fvalue = float(v)
    if fvalue < 0:
        raise argparse.ArgumentTypeError("{} is an invalid. Only positive values!" .format(fvalue))
    return fvalue

def int_positive(v):
    ivalue = int(v)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("{} is an invalid. Only positive values!" .format(ivalue))
    return ivalue

def check_solver(solver):
    strvalue = str(solver)
    allowed_values = ['SGN', 'SGD']
    if strvalue not in allowed_values:
        raise argparse.ArgumentTypeError("{} is an invalid solver. Please choose among {}!" .format(strvalue, allowed_values))
    return strvalue

def check_scheduler(scheduler):
    strvalue = str(scheduler)
    allowed_values = ['const', 'linear', 'step', 'cos']
    if strvalue not in allowed_values:
        raise argparse.ArgumentTypeError("{} is an invalid scheduler. Please choose among {}!" .format(strvalue, allowed_values))
    return strvalue

def check_rho_schedule(scheduler):
    strvalue = str(scheduler)
    allowed_values = ['const', 'linear', 'cos']
    if strvalue not in allowed_values:
        raise argparse.ArgumentTypeError("{} is an invalid scheduler. Please choose among {}!" .format(strvalue, allowed_values))
    return strvalue

def normalize_meanstd(a, axis=None):
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = numpy.mean(a, axis=axis, keepdims=True)
    std = numpy.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

def check_dataset(dataset):
    strvalue = str(dataset)
    allowed_values = ['cifar10', 'fashion', 'mnist']
    if strvalue not in allowed_values:
        raise argparse.ArgumentTypeError("{} is an invalid dataset. Please choose among {}!" .format(strvalue, allowed_values))
    return strvalue

def check_dataset_mlp(dataset):
    strvalue = str(dataset)
    allowed_values = ['boston', 'sine', 'mnist']
    if strvalue not in allowed_values:
        raise argparse.ArgumentTypeError("{} is an invalid dataset. Please choose among {}!" .format(strvalue, allowed_values))
    return strvalue

