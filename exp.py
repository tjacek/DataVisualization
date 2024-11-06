import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
import json
import gc
import dataset,autoencoder,deep,ensemble,utils

class Exp(object):
    def __init__(self,
                 feats:list,
                 clfs:list,
                 n_splits:int,
                 n_repeats:int,
                 aggr:bool,
                 input_paths:list,
                 output_path:str):
        self.feats=feats
        self.clfs=clfs
        self.n_splits=n_splits
        self.n_repeats=n_repeats
        self.aggr=aggr
        self.input_paths=input_paths
        self.output_path=output_path

    def iter(self):
        protocol=self.get_protocol()
        clf_iterator=self.get_clf_iterator()
        for path in self.input_paths:
            data_id=path.split('/')[-1]
            data=dataset.read_csv(path)
            splits=protocol.get_split(data)
            for feat_type_i in self.feats:
                 feat_i=get_features(feat_type_i)
                 data_i=feat_i(data)
                 for clf_type_j,clf_j in clf_iterator(data_i): 
                    desc=(data_id,feat_type_i,clf_type_j)
                    for split_k in splits:
                        result=split_k.eval(data_i,clf_j())    
                        gc.collect()
                        yield desc,result
    
    def save(self):
        for i,(desc_i,result_i) in enumerate(self.iter()):
            out_i=prepare_path(self.output_path,desc_i)
            out_i=f"{out_i}/{i}"
            print(out_i)
            result_i.save(out_i)        
        with open(f"{self.output_path}/desc", 'w') as f:
            json.dump(self.get_desc(), f)

    def get_protocol(self):
        if(self.aggr):
             return AggrSplit(self.n_splits,self.n_repeats)
        else:
             return UnaggrSplit(self.n_splits,self.n_repeats)

    def get_clf_iterator(self):
        clfs,data_clfs={},{}
        for clf_type in self.clfs:
            is_data,clf=get_clf(clf_type)
            if(is_data):
                data_clfs[clf_type]=clf
            else:
                clfs[clf_type]=clf
        def helper(data):
            for clf_type_i,clf_i in clfs.items():
                yield clf_type_i,clf_i
            for clf_type_i,clf_i in data_clfs.items():
                yield clf_type_i,clf_i(data) 
        return helper

    def get_desc(self):
        return { "feats":self.feats,"clf":self.clfs,"n_splits":self.n_splits,
                 "n_repeats":self.n_repeats,"aggr":self.aggr}

class UnaggrSplit(object):
    def __init__(self,n_splits,n_repeats):
        self.n_splits=n_splits
        self.n_repeats=n_repeats

    def get_split(self,data):
        rskf=RepeatedStratifiedKFold(n_repeats=self.n_repeats, 
                                     n_splits=self.n_splits, 
                                     random_state=0)
        splits=[]
        for train_index,test_index in rskf.split(data.X,data.y):
            splits.append(self.Split(train_index,test_index))
        return splits

    class Split(object):
        def __init__(self,train_index,test_index):
            self.train_index=train_index
            self.test_index=test_index

        def eval(self,data,clf):
            return data.eval(train_index=self.train_index,
                             test_index=self.test_index,
                             clf=clf,
                             as_result=True)

class AggrSplit(object):
    def __init__(self,n_splits,n_repeats):
        self.n_splits=n_splits
        self.n_repeats=n_repeats

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
                                        clf=clf,
                                        as_result=False)
                all_pred.append(pred_t)
                all_test.append(test_t)
            all_pred=np.concatenate(all_pred)
            all_test=np.concatenate(all_test)
            return dataset.Result(all_pred,all_test)

def get_clf(clf_type):
    if(clf_type=="RF"): 
        return False,lambda : RandomForestClassifier(class_weight="balanced")#_subsample")
    if(clf_type=="LR"):
        return False,lambda : LogisticRegression(solver='liblinear')
    if(clf_type=="deep"):
        return True, deep.DeepFactory
    if(clf_type=="gauss_ens"):
        return True,ensemble.GEnsembleFactory
    if(clf_type== "class_ens"):
        return True,ensemble.CEnsembleFactory
    raise Exception(f"Unknow clf type:{clf_type}")

def get_features(feat_type):
    if(feat_type=="antr"):
        return autoencoder.AthroFeatures()
    if(feat_type=="base"):
        return lambda x:x
    if(feat_type=="deep"):
        return deep.DeepFeatures()
    raise Exception(f"Unknow feature type:{feat_type}")

def prepare_path(out_path,desc):
    out_i=out_path
    for desc_i in desc:
        utils.make_dir(out_i)
        out_i=f'{out_i}/{desc_i}'
    utils.make_dir(out_i)
    return out_i

def read_conf(in_path):
    with open(in_path, 'r') as file:
        data = json.load(file)
        return data

def build_exp(in_path:str):
    conf=read_conf(in_path)
    if("data" in conf):
        data_dir=conf["data_dir"]
        input_paths=[ f"{data_dir}/{path_i}"
                       for path_i in conf["data"]]
    else:
        input_paths=utils.top_files(conf["data_dir"])
    return Exp(feats=conf["feats"],
               clfs=conf["clfs"],
               n_splits=conf["n_splits"],
               n_repeats=conf["n_repeats"],
               aggr=conf["aggr"],
               input_paths=input_paths,
               output_path=conf["output_path"] 
            ) 

if __name__ == '__main__':
    exp=build_exp("json/deep2.js")
    exp.save()