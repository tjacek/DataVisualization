import os
import sys
import time

import numpy

import theano
import theano.tensor as T
import arff

#class DeepClassifer(object):
#    def learn(self,data):
#        return None
#
#    def classifi(object):
#        return None

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out=20):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]


    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def getLearnFunction(dataX,dataY,n_in,n_out,learning_rate ):
    x = T.matrix('x')  
    y = T.ivector('y') 
    classifier=LogisticRegression(x,n_in, n_out)
    cost = classifier.negative_log_likelihood(y)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    theano.printing.pprint(g_W)
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]
    train_model = theano.function(
        inputs=[x,y],
        outputs=cost,
        updates=updates,
       # givens={
       #     x: dataX,
       #     y: dataY
       # }
        name = "train"
    )
    #theano.printing.debugprint(train_model)
    #theano.printing.pydotprint(train_model, outfile="img.png", var_with_name_simple=True)
    return train_model

def trainLR(n_epochs=1000,learning_rate =0.13):
    name="linearHist.arff"
    dataset=arff.readArffDataset(name)
    dataX=dataset.data
    dataY=dataset.target  
    print(dataX.shape)
    print(dataY.shape)
    n_in=len(dataX[0])
    n_out=20
    learn=getLearnFunction(dataX,dataY,n_in,n_out,learning_rate)
    epoch=0
    while (epoch < n_epochs):
        print(learn(dataX,dataY))
        epoch+=1

trainLR()
