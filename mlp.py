import os
import sys
import time

import numpy

import theano
import theano.tensor as T
import arff
import sklearn.cross_validation as cv
from deep import LogisticRegression
from sklearn import preprocessing
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input
        
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]


class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):

        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors


        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def getModel(dataX,dataY,n_in,n_out,learning_rate=0.01,
       L1_reg=0.00, L2_reg=0.0001, n_hidden=500):
    x = T.matrix('x')  
    y = T.ivector('y') 

    rng = numpy.random.RandomState(1234)
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        n_hidden=n_hidden,
        n_out=n_out
    )

    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    validate_model = theano.function(
        inputs=[x,y],
        outputs=classifier.errors(y),
    )

    gparams = [T.grad(cost, param) for param in classifier.params]

    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    train_model = theano.function(
        inputs=[x,y],
        outputs=cost,
        updates=updates
    )

    return classifier,train_model,validate_model 

def trainLoop(train_x,train_y,n_epochs=2000,learning_rate =0.13):
    print("Test")
    n_in=len(train_x[0])
    n_out=20
    cls,learn,validate=getModel(train_x,train_y,n_in,n_out,learning_rate)
    epoch=0
    err_min=0.02
    while (epoch < n_epochs):
        learn(train_x,train_y)
        err=validate(train_x,train_y)
        print(err)
        if(err==0.0):
            break
        epoch+=1
    return cls,validate

def trainLR(name="linearHist.arff"):
    dataset=arff.readArffDataset(name)
    X=dataset.data
    Y=dataset.target
    train_x, test_x, train_y, test_y = cv.train_test_split(
                                       X, Y, test_size=0.5, random_state=0)  
    cls,validate=trainLoop(train_x,train_y)
    print("...........................\n")
    print(validate(test_x,test_y))

if __name__ == "__main__":
    trainLR()
