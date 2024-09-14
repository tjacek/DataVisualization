from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from keras import regularizers
import dataset

class AthroFeatures(object):
    def __init__(self,mult=2):
        self.mult=mult

    def build(self,data):
        n_dim=data.dim()
        input_layer = Input(shape=n_dim)
        nn=input_layer
        nn=Dense(self.mult*n_dim,
        	     activation='relu',
                 name=f"hidden",
                 activity_regularizer=regularizers.L1(1.0))(nn)
        nn=Dense(n_dim,activation='relu')(nn)
        return Model(inputs=input_layer, 
	                 outputs=nn)

if __name__ == '__main__':
    data=dataset.read_csv("uci/cleveland")
    anthr=AthroFeatures()
    model=anthr.build(data)
    model.summary()