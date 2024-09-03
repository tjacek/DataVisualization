import os.path
from functools import wraps

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def top_files(path):
    if(type(path)==str):
        paths=[ f'{path}/{file_i}' for file_i in os.listdir(path)]
    else:
        paths=path
    paths=sorted(paths)
    return paths


def get_pca(X,y=None):
    pca = PCA()#n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    return Dataset(X=pca.transform(X),
                y=y)

class DirFun(object):
    def __init__(self,dir_args=None):
        if(dir_args is None):
            dir_args=[("in_path",0)]
        self.dir_args=dir_args

    def __call__(self, fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            args=list(args)
            in_path=self.get_input(*args, **kwargs)
            if(not os.path.isdir(in_path) ):
                return fun(*args, **kwargs)
            result_dict={}
            for in_i in top_files(in_path):
                id_i=in_i.split('/')[-1]
                args[0]=in_i
                result_i=fun(*args, **kwargs)
                result_dict[id_i]=result_i
            return result_dict
        return decor_fun
        
    def get_input(self,*args, **kwargs):
        name,i=self.dir_args[0]
        if(name in kwargs):
            return kwargs[name]
        return args[0]