import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
import utils

class HMGenerator(object):
    def __init__(self,fun,feats=None,distance=True):
        self.fun=fun
        self.feats=None
        self.distance=True

    def __call__(self,in_path,out_path):
        utils.make_dir(out_path)
        @utils.DirFun({'in_path':0,'out_path':1})
        def helper(in_path,out_path):
            matrix=self.fun(in_path)
            if(self.distance): 
                 show_distance(matrix,show=False)
            else:
                 show_matrix(matrix,show=False)
            plt.savefig(out_path)
        helper(in_path,out_path)

def show_matrix(matrix,show=True):
    matrix=np.array(matrix)
    matrix[np.isnan(matrix)]=0.0
    matrix[matrix==np.inf]=0
    matrix/= np.sum(matrix)
    matrix= 10*matrix
    matrix=np.around(matrix,decimals=2)
    plt.figure(figsize = (10,7))
    sns.heatmap(matrix, annot=True)
    if(show):
        plt.show()

def show_distance(matrix,show=True):
    mds=MDS(n_components=2,
                     dissimilarity="precomputed")
    pos=mds.fit(matrix).embedding_
    plt.figure()
    for i,pos_i in enumerate( pos):
        print(pos_i.shape)
        plt.text(pos_i[0], 
                 pos_i[1], 
                 str(i))
    pos_min=pos.min()
    pos_max=pos.max()
    plt.xlim((pos_min,pos_max))
    plt.ylim((pos_min,pos_max))
    if(show):
        plt.show()

def bar_plot(x):
    for i,value in enumerate(x):
        plt.bar(i, width=0.8, 
            height=value,
            bottom=None, 
            align='center', data=None)
    plt.show()

def stacked_bar_plot(hist,show=True):
    n_clusters=hist.shape[0]
    clusters= np.arange(n_clusters)
    hist_dict={i:hist_i for i,hist_i in enumerate(hist.T)}
    width = 0.6 
    fig, ax = plt.subplots()
    bottom = np.zeros(n_clusters)
    for i,hist_i in hist_dict.items():
        print(hist_i.shape)
        p = plt.bar(clusters, 
                    hist_i, 
                    width=width, 
                    label=i, 
                    bottom=bottom)
        bottom += hist_i
        ax.bar_label(p, label_type='center')
    if(show==True):
        plt.show()
    if(type(show)==str):
        plt.savefig(show)

if __name__ == '__main__':
    a=np.ones((10,10))-np.identity(10)
    show_matrix(a)