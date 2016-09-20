import theano
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import softmax
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

def softmax_3d(z_v):
    #Return a 3d vector for the input, instead of just usual 2d
    pass

class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size, timestep_n):
        self.inpt = inpt.reshape((mini_batch_size, timestep_n, self.n_in))
        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)
        #self.y_out = T.argmax(self.output, axis=1)

class SoftmaxLayer(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, mini_batch_size, timestep_n):

        #Reshape 3d to 2d so we can softmax correctly
        self.inpt = inpt.reshape((mini_batch_size*timestep_n, self.n_in))

        #The wx+b changes our 2d input to be the correct output shape
        self.inpt = softmax(T.dot(self.inpt, self.w) + self.b)

        #Finally, now that we have the correct output shape, we 
        #Convert back to 3d, making sure to use self.n_out, since this is the output
        #And it's already correctly shaped, just in 2d.
        self.output = self.inpt.reshape((mini_batch_size, timestep_n, self.n_out))
