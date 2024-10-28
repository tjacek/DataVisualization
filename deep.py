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
        output= self.model.get_layer("layer_2").output 
        extractor=Model(inputs=self.model.input,
                        outputs=output)
        new_X=extractor.predict(data.X)
        return dataset.Dataset(X=new_X,
                               y=data.y)
class ClfCNN(object):
    def __init__(self,n_epochs=1000,
                      default_cats=None,
                      default_weights=None,
                      verbose=0):
        self.n_epochs=n_epochs
        self.default_cats=default_cats
        self.default_weights=default_weights
        self.model=None
        self.verbose=verbose

    def fit(self,X,y):
        params={'dims': X.shape[1],
                'n_epochs':self.n_epochs}
        if(self.default_cats is None):
            params['n_cats']= int(max(y))+1
        else:
            params['n_cats']=self.default_cats   
        self.model=make_nn(params)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy']) 
        if(self.default_weights is None):
            self.default_weights=dataset.get_class_weights(y)
        y=tf.one_hot(y,
                     depth=params['n_cats'])
        self.model.fit(x=X,
                       y=y,
                       epochs=params['n_epochs'],
                       class_weight=self.default_weights,
                       callbacks=get_callback(),
                       verbose=self.verbose)

    def predict(self,X):
        pred= self.model.predict(X,verbose=self.verbose)
        return np.argmax(pred,axis=1)
    
    def predict_proba(self,X):
        return self.model.predict(X,verbose=self.verbose)

def train(data,n_epochs=1000,report=True):
    params={"n_cats":data.n_cats(),
            'dims':data.dim(),
            'n_epochs':n_epochs}
    model=make_nn(params)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam') 
    skf=StratifiedKFold(n_splits=2, 
                        shuffle=True, 
                        random_state=None)
    train,test= list(skf.split(data.X,data.y))[0]
    (X_train,y_train),(X_test,y_test)=data.split(train,test)
    y_train=tf.one_hot(y_train,depth=params['n_cats'])
    model.fit(X_train,y_train,
            class_weight=data.class_weight())
    if(report):
        y_pred=cnn.predict(X_test)
        y_pred=np.argmax(y_pred,axis=1)
        y_test=np.argmax(y_test,axis=1)
        print(classification_report(y_test, y_pred))
    return model

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
    return tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                            patience=15)

if __name__ == '__main__':
    data=dataset.read_csv("uci/cleveland")
    train(data,report=True)