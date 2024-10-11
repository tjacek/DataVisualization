import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils

class HMGenerator(object):
    def __init__(self,fun,feats=None):
        self.fun=fun
        self.feats=None

    def __call__(self,in_path,out_path):
        utils.make_dir(out_path)
        @utils.DirFun({'in_path':0,'out_path':1})
        def helper(in_path,out_path):
            matrix=self.fun(in_path)
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

if __name__ == '__main__':
    a=np.ones((10,10))-np.identity(10)
    show_matrix(a)