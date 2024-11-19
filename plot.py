import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import dataset,reduction,utils
import deep,clustering,density 

def reduce_plots(data,out_path=None,transform=None,show=False):
    if(type(data)==str):
        data=dataset.read_csv(data)
    color_helper= make_color_map(data)
    if(out_path):
        utils.make_dir(out_path)
    for transform_type_i,data_i in transorm_iter(transform):
        print(transform_type_i)
        text_plot(data=data_i,
                  cmap=color_helper,
                  labels=data_i.y,
                  title=transform_type_i,
                  comment="")
        if(out_path):
            plt.savefig(f'{out_path}/{transform_type_i}')

@utils.DirFun({'in_path':0,'out_path':1})
def clustering_plots(in_path,
                     out_path=None,
                     transform=None,
                     clust_type="gauss"):
    data=dataset.read_csv(in_path)
    clust=clustering.get_clustering(clust_type)(data)
    n_clusters=clust.n_clusters()
    color_helper= make_color_map(n_clusters)
    if(out_path):
        utils.make_dir(out_path)
    for transform_type_i,data_i in transorm_iter(data,transform):
        print(transform_type_i)
        text_plot(data=data_i,
                  cmap=color_helper,
                  labels=clust.cls_indices,
                  title=transform_type_i,
                  comment=f"Number of clusters:{n_clusters}")
        if(out_path):
            plt.savefig(f'{out_path}/{transform_type_i}')

@utils.DirFun({'in_path':0,'out_path':1})
def nn_plots(in_path,
             out_path=None,
             transform=None,
             k=10):
    data=dataset.read_csv(in_path)
    same_class=density.near_density(in_path=data,
                                    k=k,
                                    all_cats=True)
    same_class*=5
    same_class=same_class.astype(int)
    col=['b','y','g','grey','r']
    cmap = colors.ListedColormap(col)
    utils.make_dir(out_path)
    for transform_type_i,data_i in transorm_iter(data,transform):
        print(transform_type_i)
        text_plot(data=data_i,
                  cmap=cmap,
                  labels=same_class,
                  title=transform_type_i,
                  comment=f"k={k}")
        if(out_path):
            plt.savefig(f'{out_path}/{transform_type_i}')
def transorm_iter(data,transform):
    if(transform is None):
        transform=["pca","spectral","lda","lle","mda","tsne","ensemble"]
    for transform_type_i in transform:
        transform_fun=reduction.get_reduction(transform_type_i) 
        data_i=transform_fun(data,n_components=2)
        yield transform_type_i,data_i

def error_plot(params,transform=None):
    if(transform is None):
        transform=["pca","spectral","lda","lle","mda","tsne","ensemble"]
    transform_fun={ transform_i:reduction.get_reduction(transform_i) 
                       for transform_i in transform}
    col,com=['b','y','g','r'],['neither',params['first'],params['second'],'both']
    cmap = colors.ListedColormap(col)
    text=[ f'{color_i}-{com_i}' 
            for color_i,com_i in  zip(col,com)]
    text=",".join(text)
    out_path=params["out_path"]
    utils.make_dir(out_path)
    for data_id,data_i,comp_i in error_data(params):
        utils.make_dir(f"{out_path}/{data_id}")
        print(data_id)
        for name_j,fun_j in transform_fun.items():
            data_j=fun_j(data_i,n_components=2)
            text_plot(data_j,cmap,comp_i,title=data_id,comment=text)
            plt.savefig(f'{out_path}/{data_id}/{name_j}')

def error_data(params):
    result,feats=params["result"],params["feats"]
    comp_data=[]
    for data_path_i in utils.top_files(params['data']):
        data_i=dataset.read_csv(data_path_i)
        data_id=data_path_i.split('/')[-1]
        path_i=f'{result}/{data_id}/{feats}'
        first="%s/%s" % (path_i,params["first"])
        second="%s/%s" % (path_i,params["second"])
        comp_i=dataset.compare_results(first_path=utils.top_files(first)[0],
                                       second_path=utils.top_files(second)[0])
        comp_data.append((data_id,data_i,comp_i))
    return comp_data

def text_plot(data,cmap,labels=None,title="",comment=""):
    if(data.dim()!=2):
        raise Exception(f"dim of data:{data.dim()}")  
    if(labels is None):
        labels=data.y
    plt.figure()
    plt.title(title)
    ax = plt.subplot(111)
    for i,y_i in enumerate(data.y):
        plt.text(data.X[i, 0], 
                 data.X[i, 1], 
                 str(int(y_i)),
                 color=cmap(labels[i]),
                 fontdict={'weight': 'bold', 'size': 9})
        norm_plot(data)
    plt.figtext(0.5, 0.01, comment)

def norm_plot(data):
    x_min,y_min=data.min()
    x_max,y_max=data.max()
    plt.xlim((x_min,x_max))
    plt.ylim((y_min,y_max))

def make_color_map(n_cats):
    if(type(n_cats)==dataset.Dataset):
        n_cats=n_cats.n_cats()
    cat2col= np.arange(n_cats)
    np.random.shuffle(cat2col)
    def color_helper(i):
        return plt.cm.tab20(cat2col[int(i)])
    return color_helper

if __name__ == '__main__':
    nn_plots("../uci",out_path="nn_plot")
#    params={"data":"../uci",
#            "result":"uci_exp/aggr_gauss",
#            "feats":"base",
#            "first":"RF",
#            "second":"class_ens",
#            "out_path":"pl"}
#    error_plot(params)