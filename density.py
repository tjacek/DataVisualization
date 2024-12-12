import numpy as np
import pandas as pd 
import os,matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import BallTree
from scipy.special import kl_div
import seaborn as sns
import argparse,json
import dataset,utils

class PurityData(object):
    def __init__(self,cats):
        self.cats=cats

    def stats(self,type="mean",single=False):
        if(type=="median"):
            stat_fun=np.median
        if(type=="mean"):
            stat_fun=np.mean
        if(single):
            points=sum(self.cats,[])
            return stat_fun(points)
        return [ stat_fun(cat_i) for cat_i in self.cats]

    def density(self,type="mean",single=False):
        stat_purity=self.stats(type=type,single=single)
        if(single):
            return compute_density(stat_purity,x=None,show=False,n_steps=100)
        else:
            return [ compute_density(stat_purity,x=None,show=False,n_steps=100)
                       for cat_i in stat_purity]

def basic_stats(vector):
    return [ stat_i(vector)
        for stat_i in [np.mean,np.median,np.amin,np.amax]] 

def cats_by_purity(data_path,out_path):
    @utils.DirFun({'in_path':0})
    def helper(in_path):
        data_i = dataset.read_csv(in_path)
        purity_i = PurityData(knn_purity(data_i))
        raw_purity=purity_i.stats("mean")
        return np.argsort(raw_purity).tolist()
    purity_dict=helper(data_path)
    purity_dict={ name_i.split("/")[-1]:value_i 
            for name_i,value_i in purity_dict.items()}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(purity_dict, f, ensure_ascii=False, indent=4)


def purity_dataset(data_path,out_path=None):
    @utils.DirFun({'in_path':0})
    def helper(in_path):
        data_i = dataset.read_csv(in_path)
        purity_i = PurityData(knn_purity(data_i))
        raw_purity=purity_i.stats("mean")
        percent_i= list(data_i.class_percent().values())
        features=basic_stats(raw_purity)
        features+=basic_stats(percent_i)
        return features
    purity_dict=helper(data_path)
    lines=[]
    for name_i,purity_i in purity_dict.items():
        id_i=name_i.split('/')[-1]
        lines.append([id_i]+purity_i)
    cols= utils.cross(["purity_","percent_"],
                      ["mean","median","min","max"])
    df=pd.DataFrame.from_records(lines,columns= ["data"]+cols)
    if(out_path):
        df.to_csv(out_path)

def knn_purity(data,k=10):
    tree=BallTree(data.X)
    indces= tree.query(data.X,
                       k=k+1,
                       return_distance=False)
    purity=[[] for _ in range(data.n_cats())]
    for i,ind_i in enumerate(indces):
        y_i=data.y[i]
        near=[ int(y_i==data.y[ind_j]) for ind_j in ind_i[1:]]
        purity[int(y_i)].append(np.mean(near))
    return purity

def compute_density(value,x=None,show=False,n_steps=100):
    value=value.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(value)
    if(x is None):
        a_max,a_min=np.max(value),np.min(value)
        delta= (a_max-a_min)/n_steps
        x=np.arange(n_steps)*delta
        x-=a_min
    log_dens= kde.score_samples(x.reshape(-1, 1))
    dens=np.exp(log_dens)
    if(show):
        simple_plot(x,dens)
    return x,dens

@utils.DirFun({'in_path':0,'out_path':1})
def density_plot(in_path,out_path,k=10,all_cats=True):
    print(in_path)
    data_i = dataset.read_csv(in_path)
    purity_i = PurityData(knn_purity(data_i))
    raw_purity=purity_i.stats("mean",
                              single=(not all_cats))
    raise Exception(raw_purity)
@utils.DirFun({'in_path':0,'out_path':1})
def density_plot(in_path,out_path,k=10,all_cats=True):
    print(in_path)
    near_mean=near_density(in_path,
                           k=k,
                           all_cats=all_cats) 
    if(all_cats):
        x_order,dens= compute_density(near_mean,
                                      show=False,
                                      n_steps=100)
        fig, ax = plt.subplots()
        ax.plot(x_order,dens)
        plt.savefig(out_path)
    else:
        fig, ax = plt.subplots()
        x=np.arange(100)*0.01
        for i,near_i in enumerate(near_mean):
            _,dens_i= compute_density(value=near_i,
                                      x=x,
                                      show=False,
                                      n_steps=100)
            ax.plot(x, dens_i, label=f"{i}-{len(near_i)}")
        plt.legend()
        plt.savefig(out_path)

#def near_density(in_path,k=10,all_cats=True):
#    data=dataset.read_csv(in_path)
#    tree=BallTree(data.X)
#    indces= tree.query(data.X,
#                       k=k+1,
#                       return_distance=False)
#    same_class=[[] for _ in range(data.n_cats())]
#    for i,ind_i in enumerate(indces):
#        y_i=data.y[i]
#        near=[ int(y_i==data.y[ind_j]) for ind_j in ind_i[1:]]
#        same_class[int(y_i)].append(np.mean(near))
#    if(all_cats):
#        same_class=sum(same_class,[])
#        return np.array(same_class)
#    return [np.array(cat) for cat in same_class]



def simple_plot(x_order,dens):
    fig, ax = plt.subplots()
    ax.plot(x_order,dens)
    plt.show()

def size_plot(in_path,k=10):
    @utils.DirFun({'in_path':0})
    def helper(in_path):
        near_mean=near_density(in_path,
                               k=k,
                               all_cats=False)
        return [ (near_i.shape[0],np.median(near_i)) 
                    for near_i in enumerate(near_mean)]
    near_dict=helper(in_path)
    points=[]
    for _,x_i in near_dict.items():
        points+=x_i
    points=np.array(points)
    plt.scatter(x=points[:,0], y=points[:,1])
    plt.show()

def acc_plot(data_path,result_path,clfs="class_ens",k=10):
    near_dict=get_near_dict(data_path,k=k)
    if(type(clfs)==str):
        clfs=[clfs]    
    plt.title("Individual classes in uci datasets")
    for clf_i in clfs:
        points_i=acc_points(clf=clf_i,
                            near_dict=near_dict,
                            result_path=result_path)
        plt.scatter(x=points_i[:,0], 
                    y=points_i[:,1],
                    label=clf_i)
    plt.xlabel(f"knn (k={k})")
    plt.ylabel("Partial Acc")
    plt.legend()
    plt.show()

def diff_acc_plot(data_path,result_path,clf_pair,k=10):
    if(len(clf_pair)<2):
        raise Exception("Two clf required")
    near_dict=get_near_dict(data_path,k=k)
    first_points=acc_points(clf=clf_pair[0],
                            near_dict=near_dict,
                            result_path=result_path)
    second_points=acc_points(clf=clf_pair[1],
                             near_dict=near_dict,
                             result_path=result_path)
    x=first_points[:,0]
    y=first_points[:,1]-second_points[:,1]
    plt.title(f"Diff between {clf_pair[0]} -{clf_pair[1]}")
    plt.scatter(x=x[y<0], y=y[y<0])
    plt.scatter(x=x[y>0], y=y[y>0])
    plt.xlabel(f"knn-purity (k={k})")
    plt.ylabel("Partial Acc Diff")
    plt.show()

def get_near_dict(data_path,k=10):
    @utils.DirFun({'in_path':0})
    def nn_helper(in_path):
        near_mean=near_density(in_path,
                               k=k,
                               all_cats=False)
        return [ (i,np.median(near_i)) 
                    for i,near_i in enumerate(near_mean)]
    return nn_helper(data_path)

def acc_points(clf,near_dict,result_path):
    @utils.DirFun({'in_path':0})
    def acc_helper(in_path):
        if(not os.path.isdir(in_path)):
            return None
        path=utils.find_dirs(in_path ,clf)[0]
        results=dataset.read_result(path)
        acc=[result_i.all_partial_acc()
                for result_i in results]
        acc=np.array(acc)
        acc=np.mean(acc,axis=0)
        return acc  
    acc_dict=acc_helper(result_path)
    acc_dict={path_i.split("/")[2]:value_i  
            for path_i,value_i in acc_dict.items()
                if(not value_i is None)}
    near_dict={path_i.split("/")[2]:value_i  
            for path_i,value_i in near_dict.items()}
    points=[]
    for  key_i in acc_dict:
        for j,nn_j in near_dict[key_i]:
            acc_j=acc_dict[key_i][j]
            points.append((nn_j,acc_j))
    return np.array(points)

def build_plot(in_path):
    conf=utils.read_conf(in_path)
    if(conf["type"]=="acc"):
        acc_plot(data_path=conf["data_dir"],
                 result_path=conf["result_path"],
                 clfs=conf["clfs"],
                 k=conf["k"])
    if(conf["type"]=="acc"):
        density_plot(in_path=conf["data_dir"],
                     out_path=conf["output_path"],
                     k=conf["k"],
                     all_cats=False)
    if(conf["type"]=='data'):
#        purity_dataset
        cats_by_purity(data_path=conf['data_dir'],
                       out_path="purity.json")
    print(conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="json/purity.js")
    args = parser.parse_args()
    exp=build_plot(args.input)
#    nn_size_plot("../uci",k=10)
#    density_plot("../uci","density_cat",all_cats=False)