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


# TODO

1. generate laggy (träges) noise signal, to simulate possible real input 
done
2. implement generator as class, and output it via a known ss system as reference for the nn
.. then you should have train and test data, which can be generated with the input generator and the ss output
