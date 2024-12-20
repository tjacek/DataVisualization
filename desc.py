import numpy as np
import json
import pandas as pd 
import argparse
from sklearn.tree import DecisionTreeClassifier
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

def make_dataset(in_path,out_path=None):
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
#    print(df)
    if(out_path):
        df=df.round(decimals=4)
        df.to_csv(out_path)

def find_rules(in_path):
    df=pd.read_csv(in_path)
    raw=df.to_numpy()
    X,y=raw[:,2:-1],raw[:,-1]
    y[y!=1]=0
    y=y.astype(int)
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(X, y)
#    path=clf.decision_path(X)
    print( df.columns[2:] )
    import sklearn.tree
    from matplotlib import pyplot as plt
    sklearn.tree.plot_tree(clf, 
                           proportion=True,
                           feature_names=df.columns[2:])
    plt.show()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="rules.csv")
    parser.add_argument("--source", type=str, default="../uci")
    parser.add_argument('--make', action='store_true')
    args = parser.parse_args()
    if(args.make):
        make_dataset(in_path=args.source,
                     out_path=args.data)
    else:
        find_rules(in_path=args.data)