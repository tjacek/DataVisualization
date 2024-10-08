import numpy as np
import os
from collections import defaultdict
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import pandas as pd
import exp

class MetricDict(object):
    def __init__(self):
        self.metrics={"acc":accuracy_score,
                      "balance":balanced_accuracy_score}
        self.dicts={ metric_i:defaultdict(lambda : [])  
                        for metric_i in self.metrics}

    def __call__(self,result_dict):
        for id_i,results_i in result_dict.items():
            for name_j,metric_j in self.metrics.items():
                metric_values=[result_k.get_metric(metric_j) 
        	                for result_k in results_i]
                self.dicts[name_j][id_i]=metric_values 	
        return self

    def keys(self):
    	return self.dicts["acc"].keys()

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

def read_results(in_path:str):
    result_dict=defaultdict(lambda :[])
    for root, dir, files in os.walk(in_path):
        if(len(dir)==0):
            for file_i in files:
                result_i=exp.read_result(f"{root}/{file_i}")
                result_dict[root].append(result_i)
    return result_dict

df=basic_summary("exp")
print(df)