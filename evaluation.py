import numpy as np
import os
from collections import defaultdict
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from scipy import stats
import pandas as pd
import argparse
import itertools
import dataset,exp

class MetricDict(object):
    def __init__(self):
        self.metrics={"acc":accuracy_score,
                      "balance":balanced_accuracy_score}
        self.dicts={ metric_i:defaultdict(lambda : [])  
                        for metric_i in self.metrics}
        self.cols=["data","feat","clf","key"]

    def __call__(self,result_dict):
        for path_i,results_i in result_dict.items():
            for name_j,metric_j in self.metrics.items():
                metric_values=[100*result_k.get_metric(metric_j) 
        	                for result_k in results_i]
                self.dicts[name_j][get_id(path_i)]=metric_values 	
        return self

    def metric_name(self):
        return self.metrics.keys()

    def keys(self):
    	return self.dicts["acc"].keys()

    def key_frame(self):
        lines=[]
        for key_i in self.keys():
            line_i=key_i.split(',')
            line_i.append(key_i)
            lines.append(line_i)
        return pd.DataFrame.from_records(lines,
                                         columns=self.cols)

def basic_summary(in_path:str):
    result_dict=read_results(in_path)
    metrict_dict=MetricDict()(result_dict)
    lines=[]
    for id_i in metrict_dict.keys():
        line_i=id_i.split("/")[-3:]
        for metric_j in metrict_dict.metrics:
            values=metrict_dict.dicts[metric_j][id_i]
            line_i.append( np.mean(values))
            line_i.append( np.std(values))
        lines.append(line_i)
    df=pd.DataFrame.from_records(lines)
    return df	

def stat_test(in_path,query=None):
    result_dict=read_results(in_path)
    metrict_dict=MetricDict()(result_dict)
    cols=["id_x","id_y",]
    for metric_i in metrict_dict.metric_name():
        cols+=[f"{metric_i}_{col}" 
                    for col in ["diff","pvalue","sig"]]
    lines=[]
    kf=metrict_dict.key_frame()
    query_fun= get_query_fun(kf,query)
    for data_k in kf["data"].unique():
        valid_id= query_fun(data_k)
        for id_x,id_y in itertools.combinations(valid_id, 2):
            line=[id_x,id_y]
#            print(line)
            for metric_i in metrict_dict.metric_name():
                x_metric=metrict_dict.dicts[metric_i][id_x]
                y_metric=metrict_dict.dicts[metric_i][id_y]
                pvalue=stats.ttest_ind(x_metric,y_metric,
                                       equal_var=False)[1]
                diff=np.mean(x_metric)-np.mean(y_metric)
                line+=[diff,pvalue,pvalue<0.05]
                
            lines.append(line)
    df=pd.DataFrame.from_records(lines,
                                 columns=cols)
    return df

def get_query_fun(kf,query):
    if(query is None):
        query={"clf":"RF"}
    if(type(query)==list):
        def fun(data_k):
            return [ f"{data_k},{q_i}" for q_i in query]
        return fun
    query_str=' '.join([ f"& {col_i}=='{value_i}'" 
                           for col_i,value_i in query.items()])
    def fun(data_k):
        kf_query= kf.query(f"data=='{data_k}' {query_str}") 
        return list(kf_query['key'])
    return fun

def read_results(in_path:str):
    result_dict=defaultdict(lambda :[])
    for root, dir, files in os.walk(in_path):
        if(len(dir)==0):
            for file_i in files:
                result_i=dataset.read_result(f"{root}/{file_i}")
                result_dict[root].append(result_i)
    return result_dict

def get_id(path:str):
    return ",".join( path.split('/')[-3:])

def eval(args):
    if(args.summary):
        df=basic_summary(args.input)
        print(df)
    clfs=args.clfs.split()
    df=stat_test(args.input,[f'base,{clfs[0]}',f'base,{clfs[1]}'])
    df=df.sort_values(by=['balance_diff'])
    print(df)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="uci_exp/unaggr_gauss")
    parser.add_argument("--clfs", type=str, default="RF gauss_ens")
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()
    eval(args)