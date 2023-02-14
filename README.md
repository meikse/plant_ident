# CTT-NN
crude Neural Network approach for a dynamical model identification

# sources

this code is inspired by the work of the following source:

https://github.com/bmavkov/INN-for-Identification

using the generic tutorial from pyTorch:

https://pytorch.org/tutorials/beginner/nn_tutorial.html

### notes

main part for machine learning is the back-propagation (`backward()`)

it is a method of its parent class, the loss-function

in this case the loss-function is not a pre build loss function from pytorch, but rather a self defined cost-function.

1. make it possible for pytorch to conduct `.backward()` with it
(as far as i remember the gradients due to its derivative will be calculated theline backwards. Thus the loss does not has to fulfill any criteria besides being a rational number)

The cost function shall penalise the error between the state and its derivative.

xhat_i is the (model + integrator) output 


