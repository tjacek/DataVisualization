import numpy as np
import os
from collections import defaultdict
from sklearn.metrics import accuracy_score,balanced_accuracy_score
from scipy import stats
import pandas as pd
import itertools
import exp

class MetricDict(object):
    def __init__(self):
        self.metrics={"acc":accuracy_score,
                      "balance":balanced_accuracy_score}
        self.dicts={ metric_i:defaultdict(lambda : [])  
                        for metric_i in self.metrics}
        self.cols=["data","feat","clf","key"]

    def __call__(self,result_dict):
        for id_i,results_i in result_dict.items():
            for name_j,metric_j in self.metrics.items():
                metric_values=[result_k.get_metric(metric_j) 
        	                for result_k in results_i]
                self.dicts[name_j][id_i]=metric_values 	
        return self

    def metric_name(self):
        return self.metrics.keys()

    def keys(self):
    	return self.dicts["acc"].keys()

    def key_frame(self):
        lines=[]
        for key_i in self.keys():
            line_i=key_i.split('/')[-3:]
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

def stat_test(in_path:str):
    result_dict=read_results(in_path)
    metrict_dict=MetricDict()(result_dict)
    kf=metrict_dict.key_frame()
    for data_k in kf["data"].unique():
        kf_query=kf.query(f"data=='{data_k}' & clf=='RF'")
        valid_id=list(kf_query['key'])
        for x,y in itertools.combinations(valid_id, 2):
            id_x=get_id(x)
            id_y=get_id(y)
            line=f"{id_x},{id_y}"
            for metric_i in metrict_dict.metric_name():
                x_metric=metrict_dict.dicts[metric_i][x]
                y_metric=metrict_dict.dicts[metric_i][y]
                pvalue=stats.ttest_ind(x_metric,y_metric,
                                       equal_var=False)[1]
                diff=np.mean(x_metric)-np.mean(y_metric)
                line+=f",{metric_i}:{diff:.4f}:{pvalue:.4f},{pvalue<0.05}"
            print(line)

def read_results(in_path:str):
    result_dict=defaultdict(lambda :[])
    for root, dir, files in os.walk(in_path):
        if(len(dir)==0):
            for file_i in files:
                result_i=exp.read_result(f"{root}/{file_i}")
                result_dict[root].append(result_i)
    return result_dict

def get_id(path:str):
    return ",".join( path.split('/')[-3:])

df=stat_test("exp")
print(df)