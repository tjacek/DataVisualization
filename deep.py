import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization#,Concatenate
from tensorflow.keras import Input, Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import dataset

class CNN(object):
    def __init__(self,model,params):
        self.model=model
        self.params=params

    def fit(self,X,y):
        y=tf.one_hot(y,
                     depth=self.params['n_cats'])
        self.model.fit(x=X,
                       y=y,
                       epochs=self.params['n_epochs'])

    def predict(self,X):
        pred= self.model.predict(X)
        return np.argmax(pred,axis=1)

def train(data,n_epochs=100,report=True):
    params={"n_cats":data.n_cats(),
            'dims':data.dim(),
            'n_epochs':n_epochs}
    model=make_nn(params)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam') 
    cnn=CNN(model=model,
            params=params)
    skf=StratifiedKFold(n_splits=2, 
                        shuffle=True, 
                        random_state=None)
    train,test= list(skf.split(data.X,data.y))[0]
    cnn=data.eval(train_index=train,
                  test_index=test,
                  clf=cnn,
                  fit_only=not report)
    if(report):
        y_pred,y_true=cnn
        print(classification_report(y_true, y_pred))
    return cnn

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
    train(data,report=True)