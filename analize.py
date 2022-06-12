import numpy as np
import os.path
import pandas as pd 
import plot

def show_data(in_path,sep='\s+'):
    df = read_data(in_path)
#    df = remove_outliners(df,std_cond)
    names,X= split_frames(df)
    x_t= pca_transform(X)
    plot(names,x_t)

def read_data(in_path,sep='\s+'):
    if(os.path.isdir(in_path)):
        in_path=[f"{in_path}/{file_i}" 
            for file_i in os.listdir(in_path)
               if(file_i.endswith(".csv"))]
    if(type(in_path)==list):
        all_dfs=[ pd.read_csv(path_i,sep=sep) 
                for path_i in in_path]
        main_col=all_dfs[0].columns[0]
        df=all_dfs[0]
        for df_i in all_dfs[1:]:
            df = pd.merge(left=df,right=df_i, left_on=main_col, right_on=main_col)
        print(df)
    else:
         df = pd.read_csv(in_path,sep=sep)
    return df

def from_dict(dict_i):
    names=dict_i.keys()
    X=[dict_i[name_i] 
        for name_i in names]
    return names,X

def split_frames(df):
    vector=df.to_numpy()
    X=vector[:,1:]
    names=vector[:,0]
    return names,X

def remove_outliners(df,cond=None):
    if(cond is None):
        cond=max_cond
    id_name=df.columns[0]
    col_names=list(df.columns[1:])
    outliners=set()
    for col_i in col_names:
        result= df[cond(df[col_i])]
        outliners.update(list(result[id_name]))
        print(outliners)
    print(outliners)
    for out_i in outliners:
        df = df[df[id_name] != out_i]
    return df

def max_cond(df_col):
    return (df_col==df_col.max()) + (df_col==df_col.min()) 

def std_cond(df_col):
    col_std=df_col.std()
    return df_col.abs()> 2*col_std

if __name__ == "__main__":
    show_data("desc",sep=',')