import numpy as np
import matplotlib.pyplot as plt
import dataset,reduction

def plot(data):
    if(data.dim()!=2):
        raise Exception(f"dim of data:{data.dim()}")    
    fig, ax = plt.subplots()
    ax.scatter(data.X[:,0],data.X[:,1])
    cat2col= np.arange(20)
    np.random.shuffle(cat2col)
    if(data.labeled()):
       for i,y_i in enumerate(data.y):
           color_i=cat2col[int(y_i)]
           ax.annotate(y_i,data.X[i],
                    color=plt.cm.tab20(color_i))
    plt.show()

def reduce_plots(in_path,transform=None):
    data=dataset.read_csv(in_path)
    if(transform is None):
        transform={ "pca":dataset.get_pca,
                  "spectral":reduction.spectral_transform}
    for name_i,transform_i in transform.items():
        data_i=transform_i(data,n_components=2)
        plot(data_i)

if __name__ == '__main__':
    reduce_plots("uci/cleveland",transform=None)
#    data=dataset.read_csv("uci/cleveland")
#    data=dataset.get_pca(data,n_components=2)
#    plot(data)