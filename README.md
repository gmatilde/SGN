 [![Build Status](https://travis-ci.com/gmatilde/SGN.svg?token=ZcbUVpvXpbuS8sFYpWQG&branch=master)](https://travis-ci.com/gmatilde/SGN)
# Stochastic Generalized Gauss-Newton (SGN) Method for Traning Deep Neural Networks 
By [Matilde Gargiani](https://scholar.google.com/citations?user=gejXFzwAAAAJ&hl=en).

Albert-Ludwigs-Universität Freiburg.

### Table of Contents
0. [Introduction](#introduction)
0. [Why Theano](#why-theano)
0. [Citation](#citation)
0. [Installation](#installation)
0. [Usage](#usage)
0. [Algorithm](#algorithm)
0. [Results](#results)

### Introduction
This repository contains an efficient and flexible implementation of SGN method for training deep neural networks as described in the paper "On the Promise of the Stochastic Generalized Gauss-Newton Method for Training DNNs" (https://arxiv.org/pdf/2006.02409.pdf). 

### Why Theano  
Despite the undisputable popularity of Tensorflow and Pytorch as deep learning frameworks, only Theano allows an efficient and fully optimized computation of some operations such as Jacobian-vector, vector-Jacobian and Hessian-vector products (see the Theano awesome documentation at http://deeplearning.net/software/theano/tutorial/gradients.html). Since the numerical performance of SGN highly relies on the computational efficiency of such operations, here we go with a Theano implementation! :stuck_out_tongue_winking_eye:

### Citation
If you use this code or this method in your research, please cite:


	@article{Gargiani2020,
		author = {Matilde Gargiani and Andrea Zanelli and Moritz Diehl and Frank Hutter},
		title = {On the Promise of the Stochastic Generalized Gauss-Newton Method for Training DNNs},
		journal = {arXiv preprint},
		year = {2020}
	}
	
### Installation
0. Clone the repository on your machine.
0. Create a virtual environment for python with conda, e.g. ```conda create -n sgn_env python=3.6 anaconda```, and activate it, e.g.  ```source activate sgn_env```.
0. Access the cloned repository on your machine with ```cd <root>/SGN```.
0. Run the following command ```python setup.py install```.
0. If you have access to a gpu and want to use it to speedup the benchmarks, also run the following command ```conda install pygpu=0.7```.


**Note**

To check your installation, open a terminal from your conda environment, access the folder ``` scripts``` and type the following command: 
```python mlp.py --dataset mnist --layers 20 20 20 --solver SGN --rho 0.0001 --cg_min_iter 3 --cg_max_iter 3 --f res_mnist --verbose 1 --epochs 5```.


### Usage

The python files ```mlp.py``` and ```conv_net.py``` available in the folder ```scripts``` offer an example of how to use this package. 

### Algorithm
Please have a look at our paper https://arxiv.org/pdf/2006.02409.pdf for a full mathematical description of SGN. The current implementation includes also the possibility of using backtracking line search to automatically adjust the step-size (see Algorithm 3.1 in http://bme2.aut.ac.ir/~towhidkhah/MPC/Springer-Verlag%20Numerical%20Optimization.pdf) and/or using a trust region approach to automatically adapt the Levenberg-Marquardt regularization parameter (see Algorithm 4.1 in http://bme2.aut.ac.ir/~towhidkhah/MPC/Springer-Verlag%20Numerical%20Optimization.pdf). 

**SGN Adjustable Hyperparameters**

hyperparameter|description|default value
:------------:|:---------:|:-----------:
DAMP_RHO|Levenberg-Marquardt regularization parameter |10**-3
BETA|momentum parameter|0.0
CG_PREC|boolean for activation of diagonal preconditer|False
ALPHA_EXP|exponent for the diagonal preconditioner (in case it is active)|1
TR|boolean for activation of trust region heuristic|True
RHO_DECAY|in case the trust region heuristic is not active, this decaying schedule is used for adjusting the Levenberg-Marquardt parameter|'const'
K|final iteration for the decay schedule of the Levenberg-Marquardt parameter|10
ALPHA|step-size|1
PRINT_LEVEL|it regulates the level of verbosity|1
CG_TOL|accuracy of CG|10**-6
MAX_CG_ITERS|maximum number of CG iterations|10
LS|boolean for activation of line search method|True
C_LS|parameter for the line search|10**-4
RHO_LS|parameter for the line search|0.5
MAX_LS_ITERS|maximum number of line search iterations|10

### Results

Here you find an empirical evaluation of SGN across benchmarks. All data, includig the hyperparameter configurations used, are available in the folder ```results```. The results are obtained averaging 5 independent runs with seeds ```1```, ```2```, ```3```, ```4```, ```5``` were used respectively. 

**Boston Housing Regression with MLP**

0. Boston Housing regression task with a simple 2 layers MLP (solid lines: SGD; dashed lines: SGN):

Train Loss vs Seconds             |  Test Loss vs Seconds
:-------------------------:|:-------------------------:
![GitHub Logo](/figures/boston_loss_time.svg)  |  ![GitHub Logo](/figures/boston_testloss_time.svg)

All results and info on configurations used are available in the ```results/boston``` folder. 
The results with ```SGD, lr=1``` are available in the folder ```results/boston``` but are not included in the plots for readibility as SGD with this value of learning rate quickly diverges. 

The benchmarks were run on Intel(R) Core(TM) i7-7560U CPU @ 2.40GHz.

**Sine Wave Regression with MLP**

0. Sine wave regression task (frequency 10) with a simple 3 layers MLP (solid lines: SGD; dashed lines: SGN):

Train Loss vs Seconds             |  Test Loss vs Seconds
:-------------------------:|:-------------------------:
![GitHub Logo](/figures/sine10_loss_time.svg)  |  ![GitHub Logo](/figures/sine10_testloss_time.svg)

All results and info on configurations used are available in the ```results/sine_10``` folder. 

The results with ```SGD, lr=1``` are available in the folder ```results/sine_10``` but are not included in the plots for readibility as SGD with this value of learning rate quickly diverges. 

1. Sine wave regression task (frequency 100) with a simple 3 layers MLP (solid lines: SGD; dashed lines: SGN):

Train Loss vs Seconds             |  Test Loss vs Seconds
:-------------------------:|:-------------------------:
![GitHub Logo](/figures/sine100_loss_time.svg)  |  ![GitHub Logo](/figures/sine100_testloss_time.svg)

All results and info on configurations used are available in the ```results/sine_100``` folder. 

The results with ```SGD, lr=1``` are available in the folder ```results/sine_100``` but are not included in the plots for readibility as SGD with this value of learning rate quickly diverges.

The benchmarks were run on Intel(R) Core(TM) i7-7560U CPU @ 2.40GHz.

**MNIST Classification with MLP**

0. MNIST classification task with a simple 2 layers MLP (solid lines: SGD; dashed lines: SGN):

Train Loss vs Seconds             |  Test Accuracy vs Seconds
:-------------------------:|:-------------------------:
![GitHub Logo](/figures/mnist_loss_time.svg)  |  ![GitHub Logo](/figures/mnist_testacc_time.svg)

1. Table with test accuracies after 25 seconds of training (in parenthesis the value of learning rate and CG iterations for SGD and SGN respectively):
	
	<table>
	<tr> </th></tr>
	<tr><td>

	 algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGD (0.001)|17.2%  | 36    
	  SGD (0.01)|35.3%   | 36      
	  SGD (0.1)|71.9%    | 36      
	  SGD (1)|89.8%      | 36  


	</td><td>

	  algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGD (10)|11.6% | 36   
	  SGN (3) |92.9% | 8   
	  SGN (5) |92.8% | 5      
	  SGN (10)|91.6% | 3  


	</td></tr> </table>

2. Table with test accuracies after 100 seconds of training (in parenthesis the value of learning rate and CG iterations for SGD and SGN respectively):
	
	<table>
	<tr> </th></tr>
	<tr><td>

	 algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGD (0.001)|25.4%  | 145    
	  SGD (0.01)|58.2%   | 145      
	  SGD (0.1)|85.5%    | 144      
	  SGD (1)|93.0%      | 144  


	</td><td>

	  algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGD (10)|93.0% | 144   
	  SGN (3) |94.0% | 33   
	  SGN (5) |93.7% | 22      
	  SGN (10)|91.5% | 12  


	</td></tr> </table>
	
All results and info on configurations used are available in the ```results/mnist``` folder. The benchmarks were run on GeForce GTX TITAN X gpus.
	
**FashionMNIST Classification with VGG-type network**

0. FashionMNIST classification task with a simple VGG-type network (solid lines: SGD; dashed lines: SGN):

Train Loss vs Seconds             |  Train Loss vs Epochs
:-------------------------:|:-------------------------:
![GitHub Logo](/figures/fashion_loss_time.svg)  |  ![GitHub Logo](/figures/fashion_loss_epochs.svg)

1. Table with test accuracies after 50 seconds of training (in parenthesis the value of learning rate and CG iterations for SGD and SGN respectively):

	<table>
	<tr> </th></tr>
	<tr><td>

	 algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGD (0.001)|63.1%  | 15    
	  SGD (0.01)|82.3%   |15     
	  SGD (0.1)|86.7%    | 15      
	  SGD (1)| --      | --  


	</td><td>

	  algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGN (2)|84.5% | 3   
	  SGN (3)|85.2% | 2    
	  SGN (5)|85.6% | 1      
	  SGN (10)|85.4% | 1   


	</td></tr> </table>
	
2. Table with test accuracies after 150 seconds of training (in parenthesis the value of learning rate and CG iterations for SGD and SGN respectively):

	<table>
	<tr> </th></tr>
	<tr><td>

	 algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGD (0.001)|76.5%  | 46    
	  SGD (0.01)|85.5%   |46     
	  SGD (0.1)|88.5%    | 46      
	  SGD (1)| --      | --  


	</td><td>

	  algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGN (2)|86.2% | 10   
	  SGN (3)|87.2% | 8    
	  SGN (5)|87.8% | 5      
	  SGN (10)|86.0% | 3   


	</td></tr> </table>
	
All results and info on configurations used are available in the ```results/fashion``` folder. The benchmarks were run on GeForce GTX TITAN X gpus.
	
**CIFAR10 Classification with VGG-type network**

0. CIFAR10 classification task with a simple VGG-type network (solid lines: SGD; dashed lines: SGN):

Train Loss vs Seconds             |  Train Loss vs Epochs
:-------------------------:|:-------------------------:
![GitHub Logo](/figures/cifar10_loss_time.svg)  |  ![GitHub Logo](/figures/cifar10_loss_epochs.svg)

Test Accuracy vs Seconds             |  Test Accuracy vs Epochs
:-------------------------:|:-------------------------:
![GitHub Logo](/figures/cifar10_testacc_time.svg)  |  ![GitHub Logo](/figures/cifar10_testacc_epochs.svg)

Notice that ```SGN, lr=1``` diverges after epoch 7 but we included it in the plots for completeness.

1. Table with test accuracies after 650 seconds of training (in parenthesis the value of learning rate and CG iterations for SGD and SGN respectively):

	<table>
	<tr> </th></tr>
	<tr><td>

	 algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGD (0.001)|10.0%  | 46    
	  SGD (0.01)|19.0%   |45     
	  SGD (0.1)|63.1%    | 45      
	  SGD (1)| --      | --  


	</td><td>

	  algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGN (2)|44.9% | 10   
	  SGN (3)|49.3% | 8    
	  SGN (5)|56.9% | 5      
	  SGN (10)|64.7% | 3   


	</td></tr> </table>
	
2. Longer runs for some of the previous configurations (solid lines: SGD; dashed lines: SGN):

Train Loss vs Seconds             |  Train Loss vs Epochs
:-------------------------:|:-------------------------:
![GitHub Logo](/figures/cifar10_loss_time_longer.svg)  |  ![GitHub Logo](/figures/cifar10_loss_epochs_longer.svg)

Test Accuracy vs Seconds             |  Test Accuracy vs Epochs
:-------------------------:|:-------------------------:
![GitHub Logo](/figures/cifar10_testacc_time_longer.svg)  |  ![GitHub Logo](/figures/cifar10_testacc_epochs_longer.svg)

3. Table with test accuracies after 2000 seconds of training (in parenthesis the value of learning rate and CG iterations for SGD and SGN respectively):

      <table>
	<tr> </th></tr>
	<tr><td>

	 algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGD (0.1)|60.9%  | 135    

	</td><td>

	  algorithm|test acc | epochs							      
	  :-------:|:-------:|:-------:|
	  SGN (10)|71.0%  | 9
	
	</td></tr> </table>

	
All results and info on configurations used are available in the ```results/cifar10``` folder. The benchmarks were run on GeForce GTX TITAN X gpus.
