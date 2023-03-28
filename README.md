# plant_ident 

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

Following steps shall be conducted:

1. Generate possible input signal which contains a broad spectrum of possible dynamics :symbols:
2. Pipe input signal through a (stable) ss-system to obtain output the output signal :abcd:
3. Feed and train a Neural Network (NN) in form of a model to identify the ss-system dynamics without any physical infos :repeat:
4. Test the model with unknown I/O data, also generated like in step 1 :repeat_one:

## principle 

For testing purposes, `generator.py` creates input data for a possible ss-system and also generates output data in respect to it.
Functions and classes are stored in the `util.py` file.

```
                u(t)    ┌────────────┐        ┌────────────┐         │ x0
                ───────►│            │        │            │         ▼
                        │  neural    │        │ integrator │      ┌─────┐     x(t+1)
                x(t)    │  network   ├───────►│            ├─────►│ sum ├─────►
                ───────►│            │        │            │      └─────┘
                        └────────────┘        └────────────┘
```

## sources

:scroll: https://ieeexplore.ieee.org/abstract/document/9094226 \
:scroll: https://pytorch.org/tutorials/beginner/nn_tutorial.html 

## notes

Those notes are resulting from own experiences, which are not derived from a specific source

> if the ss is a linear continuous system, the model of the NN should contain at least one non-linearity

> more complexity of the input data provided for the training results in better robustness for the model


## TODO

0. Optimize scoping through training
1. Find and use real input and output data to ensure arbitrariness\
or 
2. Enlarge the amount of generated data (which slows down training)
