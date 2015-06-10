import time
import numpy
import sklearn.cross_validation as cv
import theano
import theano.tensor as T
import arff
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn import preprocessing

class dA(object):

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
        self.transform=None

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)
    
    def tranform(self,corruption_level=0.1):
        if(self.transform==None):
            tilde_x = self.get_corrupted_input(self.x, corruption_level)
            y = self.get_hidden_values(tilde_x)
            z = self.get_reconstructed_input(y)
            self.transform = theano.function([],z)
        return self.transform()

def getModel(x_in,visable_units, hidden_units=100,learning_rate=0.13):
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x_in,
        n_visible=visable_units,
        n_hidden=hidden_units
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.1,
        learning_rate=learning_rate
    )
    x = T.matrix('x')  
    train_da = theano.function(
        [],
        cost,
        updates=updates
    )
    return train_da,da

def error(da):
    err=(abs(da.x -da.tranform()))#**2
    n=len(err[0])
    v=numpy.ones(n) / float(n)
    return numpy.sum(err*v)/len(err)

def train(name="linearHist.arff",training_epochs=1000):
    name="linearHist.arff"
    dataset=arff.readArffDataset(name)
    X=dataset.data
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    train_da,da=getModel(X,len(X[0]),100)
    start_time = time.clock()
    for epoch in xrange(training_epochs):
        cost=train_da()
        print('Training epoch '+ str(epoch) + 'cost ' + str(cost))
    end_time = time.clock()
    training_time = (end_time - start_time)
    print("Time "+str(training_time))
    print(error(da))

if __name__ == "__main__":
    train()





