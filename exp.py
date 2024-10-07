import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from collections import defaultdict
from sklearn.metrics import accuracy_score
import json
import dataset,autoencoder

class Experiment(object):
    def __init__(self,
    	         feats:dict,
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
    
    def to_df(self):
        result_dict=defaultdict(lambda :[])
        for data,feat_type,clf_type,y_pair in self.iter():
            id_i=f'{data},{feat_type},{clf_type}'
            result_dict[id_i].append(Result(*y_pair))
        lines=[]
        for id_i,results in result_dict.items():
            line_i=id_i.split(",")
            acc=[result_j.get_acc() for result_j in results]
            line_i.append(np.mean(acc))
            line_i.append(np.std(acc))
            lines.append(line_i)
        print(lines)

    def iter(self):
        for path in self.input_paths:
            data_id=path.split('/')[-1]
            data=dataset.read_csv(path)
            splits=self.get_split(data)
            for feat_type_i,feat_i in self.feats.items():
                data_i=feat_i(data)
                for clf_type_j,clf_j in self.clfs.items(): 
                    for split_k in splits:
                        y_pair=split_k.eval(data_i,clf_j)
                        yield data_id,feat_type_i,clf_type_j,y_pair
    
    def get_split(self,data):
        rskf=RepeatedStratifiedKFold(n_repeats=self.n_repeats, 
                                       n_splits=self.n_splits, 
                                       random_state=0)
        splits=[]
        for t,(train_index,test_index) in enumerate(rskf.split(data.X,data.y)):
            if((t % self.n_splits)==0):
            	splits.append([])
            splits[-1].append((train_index,test_index))
        splits=[self.Split(indexes) for indexes in splits]
        return splits

    class Split(object):
        def __init__(self,indexes):
            self.indexes=indexes

        def eval(self,data,clf):
            all_pred,all_test=[],[]
            for train_t,test_t in self.indexes:
                pred_t,test_t=data.eval(train_index=train_t,
                                        test_index=test_t,
                                        clf=clf())
                all_pred.append(pred_t)
                all_test.append(test_t)
            all_pred=np.concatenate(all_pred)
            all_test=np.concatenate(all_test)
            return all_pred,all_test

class Result(object):
    def __init__(self,y_pred,y_true):
        self.y_pred=y_pred
        self.y_true=y_true

    def get_acc(self):
        return accuracy_score(self.y_pred,self.y_true)

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
	    return lambda : RandomForestClassifier()
    return lambda : LogisticRegression(solver='liblinear')

def get_features(feat_type):
    if(feat_type=="antr"):
        return autoencoder.AthroFeatures()
    return lambda x:x

exp=build_exp("exp.js")
exp.to_df()