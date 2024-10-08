import numpy as np
import matplotlib.pyplot as plt
import dataset,reduction,utils

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

def reduce_plots(data,out_path,transform=None):
    if(type(data)==str):
        data=dataset.read_csv(in_path)
    if(transform is None):
        transform={ "pca":reduction.pca_transform,
                    "spectral":reduction.spectral_transform,
                    "lda":reduction.lda_transform,
                    "lle":reduction.lle_transform,
                    "mda":reduction.mda_transform,
                    "tsne":reduction.tsne_transform,
                    "ensemble":reduction.ensemble_transform}
    utils.make_dir(out_path)
    for name_i,transform_i in transform.items():
        data_i=transform_i(data,n_components=2)
        plot(data_i,
             show=False)
        plt.savefig(f'{out_path}/{name_i}')

if __name__ == '__main__':
    import evaluation
#    data=evaluation.antr_features("uci/cleveland")["antr"]
    import deep
    data=dataset.read_csv("uci/cleveland")
    cnn=deep.train(data,report=False)
    data=cnn(data)
    reduce_plots(data,"nn_plot",transform=None)    
