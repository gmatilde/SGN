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
This repository contains an efficient and flexible implementation of SGN method for training deep neural networks as described in the paper "Stochastic generalized Gauss-Newton method (SGN): an efficient stochastic second-order method for deep learning" (http://arxiv.org/abs/TODO). 

**Note**

0. 

### Why Theano  
Despite the undisputable popularity of Tensorflow and Pytorch as deep learning frameworks, only Theano allows an efficient and fully optimized computation of some operations such as Jacobian-vector, vector-Jacobian and Hessian-vector products (see the Theano awesome documentation at http://deeplearning.net/software/theano/tutorial/gradients.html). Since the numerical performance of SGN highly relies on the computational efficiency of such operations, here we go with a Theano implementation! :stuck_out_tongue_winking_eye:

### Citation
If you use this code or this method in your research, please cite:


	@article{Gargiani2018,
		author = {Matilde Gargiani and Andrea Zanelli and Frank Hutter and Moritz Diehl},
		title = {Stochastic generalized Gauss-Newton method (SGN): an efficient stochastic second-order method for deep learning},
		journal = {arXiv preprint arXiv:TODO},
		year = {2018}
	}
	
### Installation
0. Clone the repository on your machine.
0. Create a virtual environment for python with conda, e.g. ```conda create -n sgn_env python=3.6 anaconda```, and activate it, e.g.  ```source activate sgn_env```.
0. Access the cloned repository on your machine with ```cd <root>/SGN```.
0. Run the following command ```python setup.py install```.
0. If you have access to a gpu and want to use it to speedup the benchmarks, also run the following command ```conda install pygpu==0.7```.


**Note**

To check your installation, open a terminal from your conda environment, access the folder ``` scripts``` and type the following command: 
```python mlp_mnist.py --layers 10 10 --solver SGD --lr 0.1 --f res_mnist --verbose 1```.


### Usage

### Algorithm
0. Visualizations of network structures (tools from [ethereon](http://ethereon.github.io/netscope/quickstart.html)):
	- [ResNet-50] (http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)
	- [ResNet-101] (http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50)
	- [ResNet-152] (http://ethereon.github.io/netscope/#/gist/d38f3e6091952b45198b)

0. Model files:
	- ~~MSR download: [link] (http://research.microsoft.com/en-us/um/people/kahe/resnet/models.zip)~~
	- OneDrive download: [link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)

### Results
**Sine Wave Function**

**MNIST**


0. Curves on ImageNet (solid lines: 1-crop val error; dashed lines: training error):
	![Training curves](https://cloud.githubusercontent.com/assets/11435359/13046277/e904c04c-d412-11e5-9260-efc5b8301e2f.jpg)

0. 1-crop validation error on ImageNet (center 224x224 crop from resized image with shorter side=256):

	model|top-1|top-5
	:---:|:---:|:---:
	[VGG-16](http://www.vlfeat.org/matconvnet/pretrained/)|[28.5%](http://www.vlfeat.org/matconvnet/pretrained/)|[9.9%](http://www.vlfeat.org/matconvnet/pretrained/)
	ResNet-50|24.7%|7.8%
	ResNet-101|23.6%|7.1%
	ResNet-152|23.0%|6.7%
	
0. 10-crop validation error on ImageNet (averaging softmax scores of 10 224x224 crops from resized image with shorter side=256), the same as those in the paper:

	model|top-1|top-5
	:---:|:---:|:---:
	ResNet-50|22.9%|6.7%
	ResNet-101|21.8%|6.1%
	ResNet-152|21.4%|5.7%

**CIFAR10**


0. Curves on ImageNet (solid lines: 1-crop val error; dashed lines: training error):
	![Training curves](https://cloud.githubusercontent.com/assets/11435359/13046277/e904c04c-d412-11e5-9260-efc5b8301e2f.jpg)

0. 1-crop validation error on ImageNet (center 224x224 crop from resized image with shorter side=256):

	model|top-1|top-5
	:---:|:---:|:---:
	[VGG-16](http://www.vlfeat.org/matconvnet/pretrained/)|[28.5%](http://www.vlfeat.org/matconvnet/pretrained/)|[9.9%](http://www.vlfeat.org/matconvnet/pretrained/)
	ResNet-50|24.7%|7.8%
	ResNet-101|23.6%|7.1%
	ResNet-152|23.0%|6.7%
	
0. 10-crop validation error on ImageNet (averaging softmax scores of 10 224x224 crops from resized image with shorter side=256), the same as those in the paper:

	model|top-1|top-5
	:---:|:---:|:---:
	ResNet-50|22.9%|6.7%
	ResNet-101|21.8%|6.1%
	ResNet-152|21.4%|5.7%
