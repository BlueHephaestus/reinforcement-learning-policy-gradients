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

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

    def set_inpt_3d(self, inpt_mini_batch_size, timesteps):
        """
        NEED TO FIX THIS TO STILL DO THE OPERATIONS CORRECTLY
        EVEN THOUGH NOW 3D INSTEAD OF 2D
        """
        pass

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

    def set_inpt(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax(T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)

    def set_inpt_3d(self, inpt, mini_batch_size, timesteps):
        """
        NEED TO FIX THIS TO STILL DO THE OPERATIONS CORRECTLY
        EVEN THOUGH NOW 3D INSTEAD OF 2D
        """
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = 
        self.y_out = T.argmax(self.output, axis=1)

