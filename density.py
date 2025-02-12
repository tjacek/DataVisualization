import numpy as np
import pandas as pd 
import os,matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import BallTree
from scipy.special import kl_div
import seaborn as sns
import argparse
import dataset,utils

class PurityDict(dict):
    def __init__(self, args=[]):
        super(PurityDict, self).__init__(args)
    
    def iter(self,fun):
        return { id_i:fun(purity_i)
                    for id_i,purity_i in self.items()}

    def enum(self):
        def helper(purity_i):
            near=purity_i.stats(type="median",single=False)
            return  [(i,near_i) 
                          for i,near_i in enumerate(near)]
        return self.iter(helper)

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

    def density(self,single=False):
        if(single):
            points=sum(self.cats,[])
            return compute_density(value=np.array(points),
                                   x=None,show=False,n_steps=100)
        else:
            x=np.arange(100)*0.01
            return [ compute_density(np.array(cat_i),
                                    x=x,show=False,n_steps=100)
                       for cat_i in self.cats]

    def sizes(self):
        return [len(cat_i) for cat_i in self.cats]

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

def get_purity_dict(in_path,k=10):
    @utils.DirFun({'in_path':0})
    def helper(in_path): 
        data_i = dataset.read_csv(in_path)
        return PurityData(knn_purity(data_i))
    purity_dict=helper(in_path)
    purity_dict=utils.to_id_dir(purity_dict,index=-1)
    return PurityDict(purity_dict)

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
def density_plot(in_path,out_path,k=10,single=True):
    print(in_path)
    data_i = dataset.read_csv(in_path)
    purity_i = PurityData(knn_purity(data_i))
    dens_i=purity_i.density(single=single)
    if(single):
        x_order,dens=dens_i
        fig, ax = plt.subplots()
        ax.plot(x_order,dens)
        plt.savefig(out_path)
    else:
        fig, ax = plt.subplots()
        for j,(x_j,dens_j) in enumerate(dens_i):
            ax.plot(x_j, dens_j, label=f"{j}-{len(x_j)}")
        plt.legend()
        plt.savefig(out_path)

def simple_plot(x_order,dens):
    fig, ax = plt.subplots()
    ax.plot(x_order,dens)
    plt.show()

def size_plot(in_path,k=10):
    def helper(purity_i):
        near=purity_i.stats(type="median",single=False)
        return list(zip( purity_i.sizes(),near))
    purity_dict= get_purity_dict(in_path,k=k)
    purity_dict=purity_dict.iter(fun=helper)
    points=[]
    for _,x_i in purity_dict.items():
        points+=x_i
    points=np.array(points)
    plt.scatter(x=points[:,0], y=points[:,1])
    plt.show()

def acc_plot(data_path,result_path,clfs="class_ens",k=10):
    purity_dict= get_purity_dict(data_path,k=k)
    near_dict=  purity_dict.enum()
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
    purity_dict= get_purity_dict(data_path,k=k)
    near_dict=  purity_dict.enum()    

    points=[acc_points(clf=clf_i,
                        near_dict=near_dict,
                        result_path=result_path)
            for clf_i in clf_pair]
    x=points[0][:,0]
    y=points[0][:,1]-points[1][:,1]
    plt.title(f"Diff between {clf_pair[0]} -{clf_pair[1]}")
    plt.scatter(x=x[y<0], y=y[y<0])
    plt.scatter(x=x[y>0], y=y[y>0])
    plt.xlabel(f"knn-purity (k={k})")
    plt.ylabel("Partial Acc Diff")
    plt.show()

def sig_plot(data_path:str,
             result_path:str,
             sig_path:str,
             clfs:list,
             scatter=False,
             k=10):
    purity_dict= get_purity_dict(data_path,k=k)
    near_dict=  purity_dict.enum()    
    first_acc=get_acc_dict(result_path,clfs[0])
    second_acc=get_acc_dict(result_path,clfs[1])
    df=pd.read_csv(sig_path)
    def helper(subset):
        points=[]
        for  key_i in subset:
            points_i=[]
            for j,nn_j in near_dict[key_i]:
                first_j=first_acc[key_i][j]
                second_j=second_acc[key_i][j]
                points_i.append((nn_j,first_j-second_j))
            points.append(points_i)
        return points
    subsets=[df[df["target"]==i]['data'].tolist()
                 for i in range(3)]
    series=[helper(sub_i) for sub_i in subsets]
    if(scatter):
        series=[ np.array(sum(sub_i,[])) for sub_i in series]
        plt.title(f"Diff between {clfs[0]}-{clfs[1]}")
        label_dict=["no_stat","better","worse"]
        for i,s_i in enumerate(series):
            plt.scatter(x=s_i[:,0],y=s_i[:,1],
                    label=label_dict[i])
        plt.xlabel(f"knn-purity (k={k})")
        plt.ylabel("Partial Acc Diff")
        plt.legend()
        plt.show()
    else:
        for subset_i,series_i in zip(subsets,series):
            fig, ax = plt.subplots()
            for j,ts_j in enumerate(series_i):
                ts_j=np.array(ts_j)
                print(type(ts_j))
                print(ts_j[:,0])
                ax.scatter(ts_j[:,0], ts_j[:,1], label=subset_i[j])
            plt.legend()
            plt.show()

def get_acc_dict(result_path,clf):
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
    return acc_dict

def acc_points(clf,near_dict,result_path):
    acc_dict=get_acc_dict(result_path,clf)
    points=[]
    for  key_i in acc_dict:
        for j,nn_j in near_dict[key_i]:
            acc_j=acc_dict[key_i][j]
            points.append((nn_j,acc_j))
    return np.array(points)

def make_plot(args):
    if(args.type=='diff'):
        diff_acc_plot(data_path=args.data,
                      result_path=args.result,
                      clf_pair=args.clfs.split(","))
    elif(args.type=="acc"):
        acc_plot(data_path=args.data,
                 result_path=args.result,
                 clfs=["RF","class_ens","deep"])
    elif(args.type=='sig'):
        sig_plot(data_path=args.data,
                 result_path=args.result,
                 sig_path=args.path,
                 clfs=args.clfs.split(","))
    elif(args.type=='density'):
        density_plot(in_path=args.data,
                     out_path=args.path,
                     single=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../uci")
    parser.add_argument("--result", type=str, default="uci_exp/aggr_gauss",)
    parser.add_argument("--clfs", type=str, default="RF,class_ens")
    parser.add_argument("--path", type=str, default="purity/sig.csv")
    parser.add_argument('--type',default='sig',
                        choices=['acc','density','diff',"sig"])
    args = parser.parse_args()
    make_plot(args)