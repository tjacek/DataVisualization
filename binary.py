import json

class BinaryExtractor(object):
    def __init__(self,prop_ids):
        self.prop_ids=prop_ids

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

make_extractor('adom')