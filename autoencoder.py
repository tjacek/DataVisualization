from keras.layers import Dense
from keras import Input, Model
from keras.models import Sequential
from keras import regularizers
import dataset,gauss

class AthroFeatures(object):
    def __init__(self,mult=2,l1=0.00001):
        self.mult=mult
        self.l1=l1
        self.model=None

    def __call__(self,data):
        self.train(data)
        output= self.model.get_layer("hidden").output 
        extractor=Model(inputs=self.model.input,
                        outputs=output)
        new_X=extractor.predict(data.X)
        
        return dataset.Dataset(X=new_X,
        	                   y=data.y)

    def train(self,data,n_epochs=1000):
        self.model=self.build(data)
        self.model.compile(loss='mse',
                  optimizer='adam')	
        self.model.fit(data.X,
        	           data.X,
        	           epochs=n_epochs)
        return self.model

    def build(self,data):
        n_dim=data.dim()
        input_layer = Input(shape=n_dim)
        nn=input_layer
        nn=Dense(self.mult*n_dim,
        	     activation='relu',
                 name="hidden",
                 activity_regularizer=regularizers.L1(self.l1))(nn)
        nn=Dense(n_dim,activation='relu')(nn)
        return Model(inputs=input_layer, 
	                 outputs=nn)

def aoi_compare(in_path):
    data=dataset.read_csv(in_path)
    anthr=AthroFeatures()
    anthr_data=anthr(data)
    gauss.fit_gauss(data,verbose=True)
    gauss.fit_gauss(anthr_data,verbose=True)

if __name__ == '__main__':
    aoi_compare("uci/cleveland")