import os.path
from functools import wraps
import time,json

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

def read_conf(in_path):
    with open(in_path, 'r') as file:
        data = json.load(file)
        return data

def cross(A, B):
    return [a+b for a in A for b in B]

class DirFun(object):
    def __init__(self,
                 dir_args=None,
                 input_arg='in_path',
                 out_arg='out_path'):
        if(dir_args is None):
            dir_args={"in_path":0}
        self.dir_args=dir_args
        self.input_arg=input_arg
        self.out_arg=out_arg

    def __call__(self, fun):
        @wraps(fun)
        def decor_fun(*args, **kwargs):
            old_values=self.get_input(*args, **kwargs)
            in_path=old_values[self.input_arg]
            if(not os.path.isdir(in_path)):
                return fun(*args, **kwargs)
            if(self.out_arg in old_values):
                make_dir(old_values[self.out_arg])
            result_dict={}
            for in_i in top_files(in_path):
                id_i=in_i.split('/')[-1]
                new_values={name_j:f"{value_j}/{id_i}"  
                    for name_j,value_j in old_values.items()}
                result_dict[in_i]=self.eval_fun(fun,new_values,args,kwargs)
            return result_dict
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

def find_dirs(in_path, name):
    filtered=[]
    for root, dirs, files in os.walk(in_path):
        if(name in root):
            filtered.append(root)
    return filtered

def to_id_dir(path_dict,index=-1):
    return { path_i.split("/")[index] :value_i 
            for path_i,value_i in path_dict.items()}