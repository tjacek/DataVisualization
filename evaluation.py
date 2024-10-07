import numpy as np
import os
from collections import defaultdict
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import pandas as pd
import exp

def basic_summary(in_path:str):
    result_dict=read_results(in_path)
    metrics=[accuracy_score,balanced_accuracy_score]
    lines=[]
    for id_i,results_i in result_dict.items():
        line_i=id_i.split("/")[-3:]
        for metric_j in metrics:
        	metric_values=[result_k.get_metric(metric_j) 
        	                for result_k in results_i]
        	line_i.append(np.mean(metric_values))
        	line_i.append(np.std(metric_values))
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