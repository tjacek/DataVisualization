from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import json
import autoencoder

class Experiment(object):
    def __init__(self,
    	         feats:list,
    	         clfs:dict,
    	         n_splits:int,
    	         n_repeats:int,
    	         input_paths:list,
    	         output_path:str):
    	self.feats=feats
    	self.clfs=clfs
    	self.n_splits=n_splits
    	self.n_repeats=n_repeats
    	self.input_paths=input_paths
    	self.output_path=output_path

def build_exp(in_path:str):
    conf=read_conf(in_path)
    n_splits,n_repeats=conf["n_splits"],conf["n_repeats"]
    clfs={ clf_type:get_clf(clf_type) for clf_type in conf["clfs"]}
    feats={ feat_type:get_features(feat_type) for feat_type in conf["feats"]}
    if("data" in conf):
        data_dir=conf["data_dir"]
        input_paths=[ f"{data_dir}/{path_i}"
                       for path_i in conf["data"]]
    else:
        input_paths=[conf["data_dir"]]
    output_path=conf["output_path"]    
    return Experiment(feats=feats,
    	              clfs=clfs,
    	              n_splits=n_splits,
    	              n_repeats=n_repeats,
    	              input_paths=input_paths,
    	              output_path=output_path)

def read_conf(in_path):
    with open(in_path, 'r') as file:
        data = json.load(file)
        return data

def get_clf(clf_type):
    if(clf_type=="RF"):	
	    return RandomForestClassifier()
    return LogisticRegression(solver='liblinear')

def get_features(feat_type):
    if(feat_type=="antr"):
        return autoencoder.AthroFeatures()
    return lambda x:x

build_exp("exp.js")
