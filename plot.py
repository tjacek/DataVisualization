import matplotlib.pyplot as plt
import dataset

def plot(data_dict):
    names,data,y=data_dict.to_dataset()
    names=data_dict.names()
    if(type(data_dict)==dataset.LabeledDataset):
        plot_supervised(names,data,y)
    else:
        plot_unsupervised(names,data)

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