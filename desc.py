import json
import dataset,density,utils

def ord_exp(in_path,out_path):
    compute_funcs={"purity":compute_purity,
                   "size":compute_size}
    utils.make_dir(out_path)
    for name_i,fun_i in compute_funcs.items():
        gen_ordering(in_path=in_path,
                     out_path=f"{out_path}/{name_i}.json",
                     fun=fun_i)

def gen_ordering(in_path,out_path,fun):
    @utils.DirFun({'in_path':0})
    def helper(in_path):
        data_i = dataset.read_csv(in_path)
        return fun(data_i)
    value_dict=helper(in_path)
    value_dict=utils.to_id_dir(value_dict)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(value_dict, f, ensure_ascii=False, indent=4)

def compute_purity(data):
    purity = density.PurityData(density.knn_purity(data))
    return purity.stats("median")

def compute_size(data):
    return list(data.class_percent().values())

#def purity_dataset(data_path,out_path=None):
#    @utils.DirFun({'in_path':0})
#    def helper(in_path):
#        data_i = dataset.read_csv(in_path)
#        purity_i = PurityData(knn_purity(data_i))
#        raw_purity=purity_i.stats("mean")
#        percent_i= list(data_i.class_percent().values())
#        features=basic_stats(raw_purity)
#        features+=basic_stats(percent_i)
#        return features
#    purity_dict=helper(data_path)
#    lines=[]
#    for name_i,purity_i in purity_dict.items():
#        id_i=name_i.split('/')[-1]
#        lines.append([id_i]+purity_i)
#    cols= utils.cross(["purity_","percent_"],
#                      ["mean","median","min","max"])
#    df=pd.DataFrame.from_records(lines,columns= ["data"]+cols)
#    if(out_path):
#        df.to_csv(out_path)

#def cats_by_purity(data_path,out_path,k=10):
#    def helper(purity_i):
#        raw_purity=purity_i.stats("mean")
#        return np.argsort(raw_purity).tolist()
#    purity_dict= get_purity_dict(data_path,k=k)
#    purity_dict=purity_dict.iter(fun=helper)   
#    print(purity_dict)
#    if(out_path):
#        with open(out_path, 'w', encoding='utf-8') as f:
#            json.dump(purity_dict, f, ensure_ascii=False, indent=4)

def basic_stats(vector):
    return [ stat_i(vector)
        for stat_i in [np.mean,np.median,np.amin,np.amax]] 

if __name__ == '__main__':
    ord_exp(in_path="../uci",out_path="ord")