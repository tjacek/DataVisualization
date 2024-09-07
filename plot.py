import matplotlib.pyplot as plt
import dataset

def plot(data):
    if(data.dim()!=2):
        raise Exception(f"dim of data:{data.dim()}")
    else:
        fig, ax = plt.subplots()
        ax.scatter(data.X[:,0],data.X[:,1])
        plt.show()

#def plot(data_dict):
#    names,data,y=data_dict.to_dataset()
#    names=data_dict.names()
#    if(type(data_dict)==dataset.LabeledDataset):
#        plot_supervised(names,data,y)
#    else:
#        plot_unsupervised(names,data)

def plot_supervised(names,X,y):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0],X[:,1])
    for i,txt in enumerate(names):
        print(txt)
        ax.annotate(txt,X[i],color=plt.cm.tab20(2*y[i]))
    plt.show()

def plot_unsupervised(names,X):
    fig, ax = plt.subplots()
    ax.scatter(X[:,0],X[:,1])
    for i,txt in enumerate(names):
        print(txt)
        ax.annotate(txt,X[i])
    plt.show()


if __name__ == '__main__':
    data=dataset.read_csv("uci/cleveland")
    data=dataset.get_pca(data,n_components=2)
    plot(data)