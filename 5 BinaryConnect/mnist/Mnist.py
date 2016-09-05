import numpy as np

import theano
import theano.tensor as T

import lasagne
import BC_utils
from lasagne import nonlinearities

from pylearn2.datasets.mnist import MNIST
from BC_utils import train_nn
from BC_layers import DenseLayer, Conv2DLayer

from collections import OrderedDict

def test_mlp(learning_rate=0.001, batch_size=100, n_hidden=500, n_hiddenLayers=3,
			n_epoches=100, verbose=True, dropout_in=0., dropout_hidden=0.,
			binary=False, stochastic=True, alpha=0.15, epsilon=1e-4):

    """
    The same meaning as the test_mlp in the hw3a.py, here the n_out and n_in are fixed.
    dropout_in:
        dropout rate of input layer
    dropout_hidden:
        dropout rate of hidden layers
    alpha & epsilon:
        parameter of the BatchNormLayers
    """
    
    print('... loading the data')

    train_set = MNIST(which_set='train', start=0, stop=50000, center=True)
    valid_set = MNIST(which_set='train', start=50000, stop=60000, center=True)
    test_set = MNIST(which_set='test', center=True)

    train_set_x = train_set.X.reshape(-1, 1, 28, 28)
    valid_set_x = valid_set.X.reshape(-1, 1, 28, 28)
    test_set_x = valid_set.X.reshape(-1, 1, 28, 28)

    train_set_x = theano.shared(np.float64(train_set_x))
    valid_set_x = theano.shared(np.float64(valid_set_x))
    test_set_x = theano.shared(np.float64(test_set_x))
 
    """
    Here the train_set_y, valid_set_y, test_set_y use one hot
    """

    train_set_y = np.hstack(train_set.y)
    valid_set_y = np.hstack(valid_set.y)
    test_set_y = np.hstack(test_set.y)

    train_set_y = np.float64(np.eye(10)[train_set_y])    
    valid_set_y = np.float64(np.eye(10)[valid_set_y])
    test_set_y = np.float64(np.eye(10)[test_set_y])

    train_set_y = theano.shared(train_set_y)
    valid_set_y = theano.shared(valid_set_y)
    test_set_y = theano.shared(test_set_y)

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]//batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]//batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]//batch_size
    

    print('... building the model')
    
    index = T.lscalar()
    x = T.tensor4('x')
    y = T.matrix('y')

    mlp = lasagne.layers.InputLayer(
    	shape=(None, 1, 28, 28),
    	input_var=x
    )

    mlp = lasagne.layers.DropoutLayer(
    	incoming=mlp,
    	p=dropout_in
    )

    for i in range(n_hiddenLayers):

    	mlp = DenseLayer(
    		incoming=mlp,
    		num_units=n_hidden,
    		binary=binary,
    		stochastic=stochastic,
            nonlinearity=nonlinearities.identity
    	)

    	mlp = lasagne.layers.BatchNormLayer(
    		incoming=mlp,
    		epsilon=epsilon,
    		alpha=alpha,
    	)

    	mlp = lasagne.layers.DropoutLayer(
    	    incoming=mlp,
    	    p=dropout_hidden
        )

    mlp = DenseLayer(
    	incoming=mlp,
    	num_units=10,
    	binary=binary,
    	stochastic=stochastic,
    	nonlinearity=nonlinearities.identity
    )

    mlp = lasagne.layers.BatchNormLayer(
    	incoming=mlp,
    	epsilon=epsilon,
    	alpha=alpha,
    )

    train_output = lasagne.layers.get_output(mlp, deterministic=False)
    train_cost = T.mean(T.sqr(T.maximum(0.,1.-y*train_output)))

    valid_output = lasagne.layers.get_output(mlp, deterministic=True)
    valid_cost = T.mean(T.sqr(T.maximum(0.,1.-y*valid_output)))
    valid_error = T.mean(T.neq(T.argmax(valid_output, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)

    test_output = lasagne.layers.get_output(mlp, deterministic=True)
    test_cost = T.mean(T.sqr(T.maximum(0.,1.-y*test_output)))
    test_error = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)
    
    if binary:
    	params = lasagne.layers.get_all_params(mlp, trainable=True)
        grads = BC_utils.gradient_calc(mlp, train_cost)
        updates = lasagne.updates.adam(loss_or_grads=grads, params=params, learning_rate=learning_rate)

    else:
    	params = lasagne.layers.get_all_params(mlp, trainable=True)
    	updates = lasagne.updates.adam(loss_or_grads=train_cost, params=params, learning_rate=learning_rate)
 
    validate_model = theano.function(
        inputs=[index],
        outputs=[valid_cost, valid_error],
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    ) 

    test_model = theano.function(
        inputs=[index],
        outputs=[test_cost, test_error],
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )


    train_model = theano.function(
        inputs=[index],
        outputs=train_cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    print('... training')

    train_nn(train_model, validate_model, test_model,
    	n_train_batches, n_valid_batches, n_test_batches, n_epoches, verbose)

if __name__=="__main__":
    test_mlp()