import numpy as np
import json
import pandas as pd 
import dataset,density,utils

def ord_exp(in_path,out_path):
    compute_funcs={"purity":compute_purity,
                   "size":compute_size}
    utils.make_dir(out_path)
    for name_i,fun_i in compute_funcs.items():
        gen_ordering( fun=fun_i,
                     in_path=in_path,
                     out_path=f"{out_path}/{name_i}.json")

def gen_ordering(fun,in_path,out_path=None):
    @utils.DirFun({'in_path':0})
    def helper(in_path):
        data_i = dataset.read_csv(in_path)
        return fun(data_i)
    value_dict=helper(in_path)
    value_dict=utils.to_id_dir(value_dict)
    if(out_path):
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(value_dict, f, ensure_ascii=False, indent=4)
    return value_dict

def compute_purity(data):
    purity = density.PurityData(density.knn_purity(data))
    return purity.stats("median")

def compute_size(data):
    return list(data.class_percent().values())

def make_dataset(in_path):
    card_funcs={"purity":compute_purity,
                "size":compute_size}
    ord_dicts=[gen_ordering(fun=fun_i,
                            in_path=in_path) 
                for _,fun_i in card_funcs.items()]
    lines=[]
    for name_i in ord_dicts[0]:
        line_i=[name_i]
        for dict_j in ord_dicts:
            line_i+=basic_stats(dict_j[name_i])
        lines.append(line_i)
    print(lines)
    cols= utils.cross(card_funcs.keys(),
                      ["_mean","_median","_min","_max"])
    print(cols)
    df=pd.DataFrame.from_records(lines,columns= ["data"]+cols)
    print(df)
#    if(out_path):
#        df.to_csv(out_path)

#def cats_by_purity(data_path,out_path,k=10):
#    def helper(purity_i):
#        raw_purity=purity_i.stats("mean")
#        return np.argsort(raw_purity).tolist()
#    purity_dict= get_purity_dict(data_path,k=k)
#    purity_dict=purity_dict.iter(fun=helper)   
#    print(purity_dict)
#    if(out_path):
#        with open(out_path, 'w', encoding='utf-8') as f:
#            json.dump(purity_dict, f, ensure_ascii=False, indent=4)

def basic_stats(vector):
    return [ stat_i(vector)
        for stat_i in [np.mean,np.median,np.amin,np.amax]] 

if __name__ == '__main__':
#    ord_exp(in_path="../uci",out_path="ord")
     make_dataset("../uci")