import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization#,Concatenate
from tensorflow.keras import Input, Model
import dataset

def train(data,n_epochs=100):
    params={"n_cats":data.n_cats(),
            'dims':data.dim()}
    model=make_nn(params)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam') 
    y_train=tf.one_hot(data.y,
                       depth=params['n_cats'])
    model.fit(data.X,
              y_train,
              epochs=n_epochs)

def make_nn(params):
    input_layer = Input(shape=(params['dims']))
    x=Dense(2*params['dims'],
              activation='relu',
              name=f"layer_1")(input_layer)
    x=Dense(params['dims'],
            activation='relu',
            name=f"layer_2")(x)
    x=Dense(params['n_cats'], 
            activation='softmax',
            name='out')(x)
    return Model(inputs=input_layer, 
                 outputs=x)

if __name__ == '__main__':
    data=dataset.read_csv("uci/cleveland")
    train(data)