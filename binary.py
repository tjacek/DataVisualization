import numpy as np
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import reduction
import analize,dataset
import plot

class BinaryExtractor(object):
    def __init__(self,prop_ids):
        self.prop_ids=prop_ids
        self.ids_prop={ value_i:key_i 
            for key_i,value_i in prop_ids.items()}

    def __len__(self):
        return len(self.prop_ids)	

    def __call__(self,instance):
        if(isinstance(instance,dict)):
            raw_dict={ key_i:self(value_i) 
                for key_i,value_i in instance.items()}
            return instance.__class__(raw_dict)	
        features=np.zeros((len(self),))
        for inst_i in instance:
            features[self.prop_ids[inst_i]]=1
        return features	
     
def make_extractor(raw_dict):
    if(type(raw_dict)==str):
        raw_dict=from_json(raw_dict)
    all_properties=set()
    for prop_i in raw_dict.values():
    	all_properties.update(prop_i)
    prop_ids={prop_i:i  for i,prop_i in enumerate(list(all_properties))}
    return BinaryExtractor(prop_ids)

def binary_transform(dict_i:dict):
    extractor= make_extractor(dict_i)
    return extractor(dict_i)

def lda_analize(data_dict):
    if(type(data_dict)==str):
        data_dict= dataset.read_class(data_dict)
    extractor= make_extractor(data_dict)
    data_dict=extractor(data_dict)
    keys,X,y= data_dict.to_dataset()
    clf=reduction.lda_transform(X,y)[1]
    for i,cof_i in enumerate(clf.coef_[0]):
        prop_i= extractor.ids_prop[i]
        print(f"{prop_i},{cof_i}")

def pca_analize(data_dict,n_eigen=2):
    if(type(data_dict)==str):
        data_dict= dataset.read_class(data_dict)
    extractor= make_extractor(data_dict)
    data_dict=extractor(data_dict)
    keys,X,y= data_dict.to_dataset()
    pca=reduction.pca_transform(X,n_dim=n_eigen)[1]
    for i in range(n_eigen):
        print("***************")
        print(pca.explained_variance_ratio_[i])
        indexes=  np.argsort(  pca.components_[i])
        for j in indexes:
            prop_j= extractor.ids_prop[j]
            cof_j= pca.components_[i][j]
            print(f"{prop_j}:{cof_j}")

#lda_analize('adom/class')

binary_dict= dataset.read_class('adom/class',binary_transform)
binary_dict=binary_dict.transform(reduction.ensemble_transform )
plot.plot(binary_dict)