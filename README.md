# plant_ident (devel)

control system plant identification through the help of a Neural Network.

(still in development!)

## prerequisites

python 3.10.6 or higher is used as well as the following external modules

| module                                                             | version |
|--------------------------------------------------------------------|---------|
| [pytorch](https://github.com/pytorch)                              | 1.13.1  |
| [python-control](https://github.com/python-control/python-control) | 0.9.2   |
| [matplotlib](https://github.com/matplotlib/matplotlib)             | 3.5.1   |

## motivation

The motivation is to feed data from a linear continuous state-space (ss) system as a plant through a pipeline with a neural network included.
output-matrix is an eye matrix for the ss-system.
This pipeline receives input as well as output data from the real ss-system which shall be analyzed.
Due to training of the neural network, the ss shall be identified without any further information about the physical system dynamics (black box approach).

## principle 

For testing purposes, `generator.py` creates input data for a possible ss-system and also generates output data in respect to it.
Functions and classes are stored in the `util.py` file.


```
                u(t)    ┌────────────┐        ┌────────────┐         │ x(t)
                ───────►│            │        │            │         ▼
                        │  neural    │        │ integrator │      ┌─────┐     x(t+1)
                x(t)    │  network   ├───────►│            ├─────►│ sum ├─────►
                ───────►│            │        │            │      └─────┘
                        └────────────┘        └────────────┘
```

## sources

* https://ieeexplore.ieee.org/abstract/document/9094226
* https://pytorch.org/tutorials/beginner/nn_tutorial.html


# TODO

1. prepare and overview train and test data, assuming this causes the bad train results
2. better logistic for the dataset, maybe call it with keys
3. illustrate plots in a few figures only
