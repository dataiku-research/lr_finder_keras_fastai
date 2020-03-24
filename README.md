# Learning Rate Finder in keras and fastai

This repository contains experiments comparing the keras and fastai implementations of the Learning Rate Finder (or Learning Rate Range Test). 

## Experiments

* Learning Rate Finder:
    * ResNet-56 on CIFAR-10 with batch size 512 and Stochastic Gradient Descent with Momentum 
    * keras:
        * Multiple runs with same initial weights
        * Multiple runs with random initial weights
    * fastai:
        * Multiple runs with same initial weights
        * Multiple runs with random initial weights

The following experiments are not included in the notebooks but can be obtained from the author upon request:
* keras:
    * Learning Rate Finder with different smoothing of the training loss
    * Learning Rate Finder using both training and validation losses
    * Learning Rate Finder with different batch sizes 

### Prerequisites

See `requirements.txt`.

* python 3.6
* tensorflow-gpu 1.14.0
* keras 2.3.1
* fastai 1.0.60

## Authors

* **Simona Maggio** - simona.maggio@dataiku.com - [simonamaggio](https://github.com/simonamaggio)



