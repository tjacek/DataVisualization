import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_matrix(matrix):
    matrix=np.array(matrix)
    matrix[np.isnan(matrix)]=0.0
    matrix[matrix==np.inf]=0
    matrix/= np.sum(matrix)
    print(np.around(matrix,decimals=2))
    plt.figure(figsize = (10,7))
    sns.heatmap(matrix, annot=True)
    plt.show()