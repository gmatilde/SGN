import os
import random
import time
import argparse
import json
import pickle
import numpy as np

import theano
import theano.tensor as T

class SGN(object):
    '''
    This class implements the SGN algorithm
    '''
    def __init__(self, X, y_hat, net, obj_fun, hyper_par=None, path='.'):

        self.path = path
        self.tot_iter = 0

        if hyper_par is None:
            hyper_par = {}

        if theano.config.floatX == 'float32':
            self.np_type_conv = np.float32
        else:
            self.np_type_conv = np.float64

        #dictionary containing the hyperparameters values
        self.hyper_par = self.__check_hyper(hyper_par)

        self.theta = net.theta
        self.N_par = net.N_par

        #function to evaluate the cost function
        self.cost = theano.function([X, y_hat], [obj_fun, net.out_sym, net.y_sym,
                                                 net.pred_sym])

        #compute gradient of loss wrt output of the network
        loss_grad = T.grad(obj_fun, net.out_sym)
        #compute gradient of loss wrt parameters theta
        net_grad = T.grad(obj_fun, net.theta)
        ########################################################################
        #moving average on the gradient (see https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9)
        beta = T.scalar('beta', dtype=theano.config.floatX)

        self.m = theano.shared(np.zeros((self.N_par, ), dtype=theano.config.floatX))

        momentum_update = [(self.m, beta*self.m+(1-beta)*net_grad)]
        self.update_m = theano.function([X, y_hat, beta], [obj_fun, net.pred_sym, net_grad], updates=momentum_update)
        ########################################################################
        #define operators: #see http://deeplearning.net/software/theano/tutorial/gradients.html for the definition of R-operator (Rop) and L-operator (Lop) and efficient computation of Hessian times vector
        self.p_k = theano.shared(np.ones((self.N_par, ), dtype=theano.config.floatX))
        JV = T.Rop(net.out_sym, self.theta, self.p_k)
        w = T.grad(T.sum(loss_grad*JV), net.out_sym)
        Hp_k = T.Lop(net.out_sym, self.theta, w)
        self.Delta_theta = theano.shared(np.zeros((self.N_par, ), dtype=theano.config.floatX), 'Delta_theta')
        self.r_k = theano.shared(np.zeros((self.N_par, ), dtype=theano.config.floatX), 'r_k')

        rho = T.scalar('rho', dtype=theano.config.floatX)

        #preconditioned Conjugate Gradient
        self.y_k = theano.shared(np.zeros((self.N_par, ), dtype=theano.config.floatX), 'y_k')

        M = T.vector('M')#diagonal matrix only

        alpha_k = (T.dot(self.r_k.T, self.y_k))/(T.dot(self.p_k.T, Hp_k) + rho*T.dot(self.p_k.T, self.p_k))
        Delta_theta_update = alpha_k*self.p_k
        rk_update = alpha_k*(Hp_k + rho*self.p_k)
        yk_new = M*(self.r_k+rk_update)
        beta_k = (T.dot((self.r_k+rk_update).T, yk_new))/(T.dot(self.r_k.T, self.y_k))
        pk_new = -yk_new + beta_k*self.p_k

        CG_updates = [(self.Delta_theta, self.Delta_theta+Delta_theta_update), (self.p_k, pk_new), (self.r_k, self.r_k+rk_update), (self.y_k, yk_new)]
        self.CG_f = theano.function([rho, X, y_hat, M], [theano.tensor.isnan(self.Delta_theta), theano.tensor.isnan(self.p_k), theano.tensor.isnan(self.r_k), theano.tensor.isnan(self.y_k)], updates=CG_updates)

        alpha = T.scalar('learning_rate', dtype=theano.config.floatX)

        update_theta = [(self.theta, self.theta+alpha*self.Delta_theta)]
        self.GN_CG = theano.function([alpha], updates = update_theta)
        ########################################################################
        #auxiliary function for line search
        m_hat = T.vector('m_hat', dtype=theano.config.floatX)
        rhs2_theano = T.dot(m_hat.T, self.Delta_theta)
        self.rhs2_f = theano.function([m_hat], rhs2_theano)
        ########################################################################
        self.damp_rho = []
        self.alpha = []
        self.max_cg_iters = []
        ########################################################################


    def __check_hyper(self,hyper_par):

        #defualt setting
        config = {
            'MAX_CG_ITERS': 10, # maximum number of conjugate gradient iterations
            'CG_TOL': 10**-6, # tolerance of conjugate gradient solutions
            'PRINT_LEVEL': 1, # print level
            'MAX_LS_ITERS': 10, # maximum number of line search iterations
            'LS': True, # boolean for activation of line search
            'CG_PREC': False, # boolean for activation of preconditioning for conjugate gradient
            'DAMP_RHO': 10**-3, # initial value for rho parameter used to regularize the Gauss-Newton Hessian approximation
            'RHO_LS': 0.5, # parameter for line search (see alg. 3.1 of the book 'Numerical Optimization' of Nocedal and Wright)
            'C_LS': 10**-4, # parameter for line search (see alg. 3.1 of the book 'Numerical Optimization' of Nocedal and Wright)
            'ALPHA': 1, # step size
            'BETA': 0, # parameter for exponential moving average of gradient
            'TR': False, # boolean for activation of trust region
            'ALPHA_EXP': 1, # parameter used for the preconditioner
            'DAMP_RHO_MIN': 0, # minimum value of rho for regularization of Hessian
       }



        rho_scheduler = {

        'RHO_DECAY': 'const',
        'K': 10 #final iteration to reach with decay
        }

        available_schedulers = ['const', 'linear', 'cos']

        config.update(hyper_par)

        if not(isinstance(config['MAX_CG_ITERS'], int)) or config['MAX_CG_ITERS']<=0:
            raise Exception('The number of iterations for conjugate gradient has to be a positive integer.')

        if config['CG_TOL']<0:
            raise Exception('The tolerace for conjugate gradient has to be positive.')

        if not(isinstance(config['PRINT_LEVEL'], int)) or config['PRINT_LEVEL']<0:
            raise Exception('The printing level is a non-negative integer parameter.')

        if not(isinstance(config['MAX_LS_ITERS'], int)) or config['MAX_LS_ITERS']<=0:
            raise Exception('The number of iterations of line search has to be a positive integer.')

        if config['DAMP_RHO']<0:
            raise Exception('The damping parameter rho has to be non-negative.')
        else:
            config['DAMP_RHO'] = self.np_type_conv(config['DAMP_RHO'])

        if config['ALPHA']<=0:
            raise Exception('The alpha parameter has to be positive.')
        else:
            config['ALPHA'] = self.np_type_conv(config['ALPHA'])

        if config['ALPHA_EXP']>1 or config['ALPHA_EXP']<0:
            raise Exception('The alpha exp parameter should be in the interval [0,1]')
        else:
            config['ALPHA_EXP'] = self.np_type_conv(config['ALPHA_EXP'])

        if config['BETA']<0:
            raise Exception('The damping parameter rho has to be non-negative.')
        else:
            config['BETA'] = self.np_type_conv(config['BETA'])

        if config['RHO_LS']<0:
            raise Exception('The parameter rho used in the line search procedure has to be non-negative.')
        else:
            config['RHO_LS'] = self.np_type_conv(config['RHO_LS'])

        if config['C_LS']<0:
            raise Exception('The parameter c used in the line search procedure has to be non-negative.')

        if not(isinstance(config['LS'], bool)):
            raise Exception('LS has to be of type boolean')

        if config['DAMP_RHO_MIN']< 0:
            raise Exception('The value of rho should be positive. ')
        else:
            config['DAMP_RHO_MIN'] = self.np_type_conv(config['DAMP_RHO_MIN'])

        if not(isinstance(config['CG_PREC'], bool)):
            raise Exception('CG_PREC has to be of type boolean')

        if not(isinstance(config['TR'], bool)):
            raise Exception('TR has to be of type boolean')
        else:
            if not(config['TR']):
                rho_scheduler.update(config)
                if rho_scheduler['RHO_DECAY'] not in available_schedulers:
                    raise Exception('The decay {} is not available. Please choose among {}.'.format(rho_scheduler['RHO_DECAY'], available_schedulers))
                if not(isinstance(rho_scheduler['K'], int)) or rho_scheduler['K']<=0:
                    raise Exception('The final iteration for decay should be a positive integer.')

                if rho_scheduler['PRINT_LEVEL']:
                    print('hyperparameter configuration for the solver \n{}'.format(hyper_par))
                return rho_scheduler
            else:
                if config['PRINT_LEVEL']:
                    print('hyperparameter configuration for the solver \n{}'.format(config))
                return config



    def step(self, input, target):

        #compute gradient
        avg_loss, pred, grad = self.update_m(input, target, self.hyper_par['BETA'])

        if self.hyper_par['PRINT_LEVEL']>=2:

            grad_norm = np.linalg.norm(grad)
            print('\n===>  gradient norm {:.2e}\n'.format(grad_norm))
        self.tot_iter += 1
        tmp = self.np_type_conv(self.tot_iter)
        m_hat = self.m.get_value()/(1-self.hyper_par['BETA']**tmp)
        #initialize Delta_theta to zero
        self.Delta_theta.set_value(np.zeros((self.N_par, ), dtype=theano.config.floatX))

        if self.hyper_par['CG_PREC']:
            #ATTENTION!! only supported for trust region regarding the rho damping
            rhoI = self.hyper_par['DAMP_RHO']*np.ones((self.N_par, ), dtype=theano.config.floatX)
            #epsI = 10**-8*np.ones((self.N_par, ), dtype=theano.config.floatX)
            M_prec = 1/((np.multiply(m_hat, m_hat)+rhoI)**self.hyper_par['ALPHA_EXP'])
        else:
            M_prec = np.ones((self.N_par, ), dtype=theano.config.floatX)

        tmp = np.multiply(M_prec, m_hat)

        #initialize p_k, r_k, y_k for preconditioned CG
        self.r_k.set_value(m_hat)
        self.y_k.set_value(tmp)
        self.p_k.set_value(-tmp)

        #conjugate gradient method
        k=0
        #initialize accuracy of cg
        cg_acc = 10

        if 'RHO_DECAY' in self.hyper_par:
            #adjust the damp rho parameter based on the scheduler selected
            if self.hyper_par['RHO_DECAY'] == 'cos':
                damp_rho = self.np_type_conv(self.hyper_par['DAMP_RHO']*np.cos(np.pi/2 * (self.tot_iter/self.hyper_par['K'])))
                if damp_rho<self.hyper_par['DAMP_RHO_MIN'] or self.tot_iter>self.hyper_par['K']:
                    damp_rho = self.hyper_par['DAMP_RHO_MIN']
            elif self.hyper_par['RHO_DECAY'] == 'linear':
                    m = (self.hyper_par['DAMP_RHO_MIN']-self.hyper_par['DAMP_RHO'])/(self.hyper_par['K']-1)
                    damp_rho = self.np_type_conv(self.hyper_par['DAMP_RHO'] + m*(self.tot_iter -1))
                    if damp_rho<self.hyper_par['DAMP_RHO_MIN']:
                        damp_rho = self.hyper_par['DAMP_RHO_MIN']
            elif self.hyper_par['RHO_DECAY'] == 'const':
                damp_rho = self.hyper_par['DAMP_RHO']

            self.damp_rho.append(damp_rho.item())

        else:
            self.damp_rho.append(self.hyper_par['DAMP_RHO'].item())

        if self.tot_iter >= self.hyper_par['K_CG']:
            max_iter = self.hyper_par['MAX_CG_ITERS']
        else:
            max_iter = self.hyper_par['MIN_CG_ITERS']

        while k<max_iter and cg_acc>self.hyper_par['CG_TOL']:

            k+=1

            if 'RHO_DECAY' in self.hyper_par:
                #adjust the damp rho parameter based on the scheduler selected
                delta_theta, pk, rk, yk = self.CG_f(damp_rho, input, target, M_prec)

            else:
                 delta_theta, pk, rk, yk = self.CG_f(self.hyper_par['DAMP_RHO'], input, target, M_prec)

            #check for nans
            if delta_theta.any() or pk.any() or rk.any() or yk.any():
                raise Exception('NaN detected! numerical instabilities are occurring in the computation.')

            cg_acc = max(abs(self.r_k.get_value()))

            if self.hyper_par['PRINT_LEVEL']>=2:

                if k==1:
                    print('\n++++++++++++++++++++++++++++++++++++++++ Conjugate Gradient ++++++++++++++++++++++++++++++++++++++++\n')
                if k>=10:
                    print('===>  iter {0}     ===>  accuracy {1:.2e}'.format(k, cg_acc))
                else:
                    print('===>  iter {0}      ===>  accuracy {1:.2e}'.format(k, cg_acc))

        alpha_value = self.hyper_par['ALPHA']

        if self.hyper_par['LS'] or self.hyper_par['TR']:
            #trust region
            #save the values of the parameters
            theta_old = self.theta.get_value()
            #update the parameters
            self.GN_CG(alpha_value)
            #evaluate again the right and left hand side
            lhs, _, _, pred = self.cost(input, target)

            if self.hyper_par['TR']: # see alg. 4.1 of the book 'Numerical Optimization' of Nocedal and Wright

                red_ratio = (lhs-avg_loss)/(0.5*(self.r_k.get_value()+grad)*self.Delta_theta.get_value()).sum()
                if red_ratio < 1/4:
                    self.hyper_par['DAMP_RHO'] = self.np_type_conv(3/2 * self.hyper_par['DAMP_RHO'])
                elif red_ratio > 3/4:
                    self.hyper_par['DAMP_RHO'] = max(self.np_type_conv(2/3 * self.hyper_par['DAMP_RHO']), self.hyper_par['DAMP_RHO_MIN'])

            #find the learning rate with backtracking line search (see alg. 3.1 of the book 'Numerical Optimization' of Nocedal and Wright)
            if self.hyper_par['LS']:

                rhs = avg_loss + alpha_value*self.hyper_par['C_LS']*self.rhs2_f(m_hat)

                iter_ls = 0
                while (lhs > rhs) and iter_ls<self.hyper_par['MAX_LS_ITERS']:
                    iter_ls += 1
                    #update alpha, the step length
                    alpha_value = self.hyper_par['RHO_LS']*alpha_value
                    self.theta.set_value(theta_old)
                    self.GN_CG(alpha_value)
                    #evaluate again the right and left hand side
                    lhs, _, _, pred = self.cost(input, target)
                    #rhs = avg_loss + alpha_value*rhs_2
                    if self.hyper_par['PRINT_LEVEL']>=3:
                        #TODO print the info for this optimization
                        if iter_ls ==1 :
                            print('\n++++++++++++++++++++++++++++++++++++++++++++ Line Search +++++++++++++++++++++++++++++++++++++++++++\n')
                        if iter_ls>=10:
                            print('===>  iter {0}    ===>  alpha {1:.4f}'.format(iter_ls, alpha_value))
                        else:
                            print('===>  iter {0}     ===>  alpha {1:.4f}'.format(iter_ls, alpha_value))

        else:

            self.GN_CG(alpha_value)

        self.alpha.append(alpha_value.item())
        self.max_cg_iters.append(max_iter)

        return avg_loss, grad, pred


    def reset_all_grad(self):

        #TODO
        print('Not Implemented')


class SGD(object):
    '''
    This class implements the SGD algorithm
    '''
    def __init__(self, X, y_hat, net, obj_fun, hyper_par=None, path='.'):

        self.path = path
        self.net = net

        if hyper_par is None:
            hyper_par = {}

        if theano.config.floatX == 'float32':
            self.np_type_conv = np.float32
        else:
            self.np_type_conv = np.float64

        #dictionary containing the hyperparameters values
        self.hyper_par = self.__check_hyper(hyper_par)

        self.theta = net.theta
        self.N_par = net.N_par

        self.cost = theano.function([X, y_hat], [obj_fun, net.out_sym, net.y_sym,
                                                 net.pred_sym])

        net_grad = T.grad(obj_fun, net.theta)

        self.v_k = theano.shared(np.zeros((self.N_par, ), dtype=theano.config.floatX))

        lr = T.scalar('lr')
        mu = T.scalar('mu')

        gradient_update = [(self.v_k, mu*self.v_k + lr*net_grad), (self.theta, self.theta - mu*self.v_k - lr*net_grad)]
        self.gradient_step = theano.function([X, y_hat, lr, mu], [obj_fun, net.pred_sym, net_grad], updates=gradient_update, allow_input_downcast=True)

        self.tot_iter = 0

    def __check_hyper(self,hyper_par):

        available_schedulers = ['const', 'step', 'cos', 'linear']

        config = {
            'SCHEDULER': 'const',
            'MOMENTUM': 0,
            'LR0': 1,
            'LR_K': 0.0001, #final learning rate
            'K': 10, #final iteration for the scheduling
            'PRINT_LEVEL': 1
        }

        step_scheduler = {
            'STEP': 2,
            'REDUCTION': 0.2
        }

        config.update(hyper_par)

        if not(isinstance(config['K'], int)) or config['K']<=0:
            raise Exception('The number of iterations for learning rate decay should be a positive integer.')

        if config['SCHEDULER'] not in available_schedulers:
            raise Exception('The scheduler {} is not available. Please choose among {}'.format(config['SCHEDULER'], available_schedulers))
        if config['LR0'] <= 0:
            raise Exception('The learning rate should be positive.')
        else:
            config['LR0'] = self.np_type_conv(config['LR0'])

        if config['LR_K'] <= 0:
            raise Exception('The learning rate should be positive.')
        if config['LR_K']>config['LR0']:
            raise Exception('The final learning rate should be smaller (or equal for constant behavior) than the initial one.')
        else:
            config['LR_K'] = self.np_type_conv(config['LR_K'])

        if config['MOMENTUM'] < 0:
            raise Exception('The momentum should be non-negative.')
        else:
            config['MOMENTUM'] = self.np_type_conv(config['MOMENTUM'])

        if config['SCHEDULER']=='step':
            step_scheduler.update(config)

            if not(isinstance(config['STEP'], int)) or config['STEP']<=0:
                raise Exception('The number of steps for linear learning rate decay should be a positive integer.')
            if config['REDUCTION']>1 or config['REDUCTION']<0:
                raise Exception('The reduction factor for linear learning rate decay should be a positive number in (0,1).')
            else:
                if step_scheduler['PRINT_LEVEL']:
                    print(step_scheduler)

                return step_scheduler
        else:
            if config['PRINT_LEVEL']:
                print(config)

            return config

    def step(self, input, target):

        self.tot_iter += 1


        if self.hyper_par['SCHEDULER']=='step':
            lr = self.hyper_par['LR0']
            if int((self.tot_iter-1)/self.hyper_par['STEP'])>=1:
                lr = self.np_type_conv((self.hyper_par['LR0']/(int((self.tot_iter-1)/self.hyper_par['STEP'])))*self.hyper_par['REDUCTION'])
                if lr<self.hyper_par['LR_K']:
                    lr = self.hyper_par['LR_K']
            #else:
                #lr = self.hyper_par['LR_K']
        elif self.hyper_par['SCHEDULER'] == 'linear':
            m = (self.hyper_par['LR_K']-self.hyper_par['LR0'])/(self.hyper_par['K']-1)
            lr = self.np_type_conv(self.hyper_par['LR0'] + m*(self.tot_iter -1))
            if lr<self.hyper_par['LR_K']:
                lr = self.hyper_par['LR_K']
        elif self.hyper_par['SCHEDULER'] == 'cos':
            lr = self.np_type_conv(self.hyper_par['LR0']*np.cos(np.pi/2 * (self.tot_iter/self.hyper_par['K'])))
            if lr<self.hyper_par['LR_K'] or self.tot_iter>self.hyper_par['K']:
                lr = self.hyper_par['LR_K']
        else:
            lr = self.hyper_par['LR0']

        loss, pred, grad = self.gradient_step(input, target, lr, self.hyper_par['MOMENTUM'])

        return loss, grad, pred


    def reset_all_grad(self):

        #TODO
        print('Not Implemented')
