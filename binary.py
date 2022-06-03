import numpy as np
import analize,dataset
import plot

class BinaryExtractor(object):
    def __init__(self,prop_ids):
        self.prop_ids=prop_ids

    def __len__(self):
        return len(self.prop_ids)	

    def __call__(self,instance):
        if(type(instance)==dict):
            raw_dict={ key_i:self(value_i) 
                for key_i,value_i in instance.items()}
            return dataset.DataDict(raw_dict)	
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

binary_dict= dataset.read_class('adom',binary_transform)

binary_dict=binary_dict.transform(analize.pca_transform )
plot.plot(binary_dict)