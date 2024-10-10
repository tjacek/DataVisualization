import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,BatchNormalization#,Concatenate
from tensorflow.keras import Input, Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import dataset

class DeepFeatures(object):
    def __init__(self,n_epochs=1000):
         self.n_epochs=n_epochs
         self.model=None

    def __call__(self,data):
        self.model=train(data,
                         n_epochs=self.n_epochs,
                         report=False)
        return self.model(data)

class CNN(object):
    def __init__(self,model,params):
        self.model=model
        self.extractor=None
        self.params=params

    def __call__(self,data):
#        self.train(data)
        output= self.model.get_layer("layer_2").output 
        extractor=Model(inputs=self.model.input,
                        outputs=output)
        new_X=extractor.predict(data.X)
        return dataset.Dataset(X=new_X,
                               y=data.y)

    def fit(self,X,y,class_weight):
        y=tf.one_hot(y,
                     depth=self.params['n_cats'])
        self.model.fit(x=X,
                       y=y,
                       epochs=self.params['n_epochs'],
                       class_weight=class_weight,
                       callbacks=get_callback())

    def predict(self,X):
        pred= self.model.predict(X)
        return np.argmax(pred,axis=1)

def train(data,n_epochs=1000,report=True):
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
    (X_train,y_train),(X_test,y_test)=data.split(train,test)
    cnn.fit(X_train,y_train,
            class_weight=data.class_weight())
    y_pred=cnn.predict(X_test)
    if(report):
        print(classification_report(y_test, y_pred))
    return cnn

def make_nn(params):
    input_layer = Input(shape=(params['dims'],))
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

def get_callback():
    return tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                            patience=5)


if __name__ == '__main__':
    data=dataset.read_csv("uci/cleveland")
    train(data,report=True)