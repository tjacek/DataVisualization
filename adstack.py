import os
import sys
import time
import arff
import numpy
import sklearn.cross_validation as cv
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn import preprocessing
from deep import LogisticRegression
from mlp import HiddenLayer
from autoencoder import dA

class SdA(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[288, 288],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.matrix('x')  
        self.y = T.ivector('y')  

        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
       
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)

        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self):
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')
        
        pretrain_fns = []
        for dA in self.dA_layers:
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
         
            fn = theano.function(
                inputs=[
                    self.x,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, learning_rate):

        gparams = T.grad(self.finetune_cost, self.params)

        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[self.x,self.y],
            outputs=self.finetune_cost,
            updates=updates,
            name='train'
        )

        valid_score = theano.function(
            [self.x,self.y],
            self.errors,
            name='valid'
        )

        return train_fn, valid_score

def getModel(dataX,dataY,n_in,n_out,learning_rate=0.01,
       L1_reg=0.00, L2_reg=0.0001, n_hidden=500):
    x = T.matrix('x')  
    y = T.ivector('y') 

    rng = numpy.random.RandomState(1234)
    classifier = SdA(
        numpy_rng=rng,
        n_ins=n_in,
        n_outs=n_out
    )
    pretrain=classifier.pretraining_functions()
    for f in pretrain:
	print("****************************\n")
        for i in xrange(1000):
            print(i)
            f(dataX)
    train_model,validate =classifier.build_finetune_functions(learning_rate)
    return classifier,train_model,validate 


def trainLoop(train_x,train_y,n_epochs=10000,learning_rate =0.10):
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
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    train_x, test_x, train_y, test_y = cv.train_test_split(
                                       X, Y, test_size=0.5, random_state=0)  
    cls,validate=trainLoop(train_x,train_y)
    print("...........................\n")
    print(1.0 -validate(test_x,test_y))

if __name__ == "__main__":
    trainLR()
