import numpy as np
import json

class BinaryExtractor(object):
    def __init__(self,prop_ids):
        self.prop_ids=prop_ids

    def __len__(self):
        return len(self.prop_ids)	

    def __call__(self,instance):
        if(type(instance)==dict):
            return { key_i:self(value_i) 
                for key_i,value_i in instance.items()}	
        features=np.zeros((len(self),))
        for inst_i in instance:
            features[self.prop_ids[inst_i]]=1
        return features	

def from_json(in_path):
    with open(in_path) as json_file:
        return json.load(json_file)
     
def make_extractor(raw_dict):
    if(type(raw_dict)==str):
        raw_dict=from_json(raw_dict)
    all_properties=set()
    for prop_i in raw_dict.values():
    	all_properties.update(prop_i)
    prop_ids={prop_i:i  for i,prop_i in enumerate(list(all_properties))}
    return BinaryExtractor(prop_ids)

raw_dict= from_json('adom')
extractor=make_extractor(raw_dict)
print(extractor(raw_dict))