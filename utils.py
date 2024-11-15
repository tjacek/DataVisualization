import os.path
from functools import wraps
import time

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

class DirFun(object):
    def __init__(self,dir_args=None,input_arg='in_path'):
        if(dir_args is None):
            dir_args={"in_path":0}
        self.dir_args=dir_args
        self.input_arg=input_arg

    def __call__(self, fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            old_values=self.get_input(*args, **kwargs)
            if(not os.path.exist(old_values[self.input_arg])):
                make_dir(old_values[self.input_arg])
            if(not os.path.isdir(old_values[self.input_arg])):
                return fun(*args, **kwargs)
            for in_i in top_files(old_values[self.input_arg]):
                id_i=in_i.split('/')[-1]
                new_values={name_j:f"{value_j}/{id_i}"  
                    for name_j,value_j in old_values.items()}
                self.eval_fun(fun,new_values,args,kwargs)
        return decor_fun
    
    def get_input(self,*args, **kwargs):
        mod_values={}
        for arg,index in self.dir_args.items():
            if(arg in kwargs):
                mod_values[arg]=kwargs[arg]
            else:
                mod_values[arg]=args[index]
        return mod_values

    def eval_fun(self,fun,new_values,args,kwargs):
        args=list(args)
        for arg_i,i in self.dir_args.items():
            if(arg_i in kwargs):
                kwargs[arg_i]=new_values[arg_i]
            else:
                args[i]=new_values[arg_i]
        return fun(*args, **kwargs)

def elapsed_time(fun):
    @wraps(fun)
    def helper(*args, **kwargs):
        start=time.time()
        value=fun(*args, **kwargs)
        end= time.time()
        print(f"Time:{end-start:.4f}")
        return value
    return helper