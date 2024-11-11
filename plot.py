import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import dataset,reduction,utils
import deep

class PlotGenerator(object):
    def __init__(self,feats=None):
        self.feats=None

    def __call__(self,in_path,out_path):
        utils.make_dir(out_path)
        @utils.DirFun({'in_path':0,'out_path':1})
        def helper(in_path,out_path):
            print(out_path)
            data=dataset.read_csv(in_path)
            return reduce_plots(data,out_path,transform=None)    
        helper(in_path,out_path)

def plot(data,show=True):
    if(data.dim()!=2):
        raise Exception(f"dim of data:{data.dim()}")    
    cat2col= np.arange(20)
    np.random.shuffle(cat2col)
    plt.figure()
    ax = plt.subplot(111)
    for i,y_i in enumerate(data.y):
        plt.text(data.X[i, 0], 
                 data.X[i, 1], 
                 str(int(y_i)),
                 color=plt.cm.tab20(int(y_i)),
                 fontdict={'weight': 'bold', 'size': 9})
    x_min,y_min=data.min()
    x_max,y_max=data.max()
    plt.xlim((x_min,x_max))
    plt.ylim((y_min,y_max))
    if(show):
        plt.show()

def reduce_plots(data,out_path=None,transform=None,show=False):
    if(type(data)==str):
        data=dataset.read_csv(data)
    if(transform is None):
        transform=["pca","spectral","lda","lle","mda","tsne","ensemble"]
    if(out_path):
        utils.make_dir(out_path)
    for transform_i in transform:
        transform_fun=reduction.get_reduction(transform_i) 
        data_i=transform_fun(data,n_components=2)
        plot(data_i,
             show=show)
        if(out_path):
            plt.savefig(f'{out_path}/{name_i}')

def simple_plot(in_path):
    reduce_plots(data=in_path,
                 out_path=None,
                 transform={"lle":reduction.lle_transform},
               show=True)

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

if __name__ == '__main__':
#    PlotGenerator()("../uci","reduction")
#     simple_plot(in_path="../uci/newthyroid")
#    error_plot()
    params={"data":"../uci",
            "result":"uci_exp/aggr_gauss",
            "feats":"base",
            "first":"RF",
            "second":"class_ens",
            "out_path":"pl"}
    error_plot(params)