import numpy as np
import os
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from collections import defaultdict
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from scipy import stats
import pandas as pd
import argparse,itertools
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
        return list(self.dicts.values())[0].keys()

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
    cols=["id",'acc_mean','acc_std','balance_mean','balance_std']
    df=pd.DataFrame.from_records(lines,
                                  columns=cols)
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

def format_sign(df):
    df['clf_y']=df['id_y'].apply(lambda str_i:str_i.split(",")[-1])
    df['data']=df['id_x'].apply(lambda str_i:str_i.split(",")[0])
    df['clf_x']=df['id_x'].apply(lambda str_i:str_i.split(",")[-1])
    df.drop(['id_x', 'id_y'], axis='columns', inplace=True)
    df=df[ ['data','clf_x','clf_y']+df.columns.tolist()[:-3]]
    df=df.round(4)
    return df

def eval(args):
    clfs=args.clfs.split(",")
    if(args.summary):
        df=basic_summary(args.input)
        df['data']=df['id'].apply(lambda str_i:str_i.split(",")[0])
        df['clf']=df['id'].apply(lambda str_i:str_i.split(",")[-1])
        df.drop(['id'], axis='columns', inplace=True)
        df=df[ ['data','clf']+df.columns.tolist()[:-2]]
        df=df.round(2)
        df = df[df['clf'].isin(clfs)] 
        return df
    if(len(clfs)>1):
        df=stat_test(args.input,[f'base,{clfs[0]}',f'base,{clfs[1]}'])
        df=df.sort_values(by=[args.sort])
        return format_sign(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="uci_exp/aggr_gauss")
    parser.add_argument("--clfs", type=str, default="RF,class_ens")
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--sort', type=str, default="balance_diff")
    args = parser.parse_args()
    df=eval(args)
    print(df)