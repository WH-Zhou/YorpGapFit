import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_blobs
import pandas as pd
from typing import List, Tuple
import os
import imageio
from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',
    "font.size": 20,
    "mathtext.fontset":'stix',
    # "font.serif": ['SimSun'],
}
rcParams.update(config)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
} 


def read_data(filename: str = None):
    merge_df = pd.read_csv(filename)
    x, y = merge_df['diameter (km)'], merge_df['Period (h)']
    x, y = np.log10(x), -np.log10(y)
    X = np.array(list(zip(x, y)))
    # plt.plot(x,y,'.')
    return X

def classify(X, k, b, gap) -> List[int]:
    upper_index = X[:, 1] > line(X, k=k, b=b, gap=gap)  # label 1
    lower_index = X[:, 1] < line(X, k=k, b=b, gap=-gap)   # label 2
    #  [0,1,1,1,0]
    labels = [-1] * len(X)
    for i in range(len(X)):
        if upper_index[i]:
            labels[i] = 1
        elif lower_index[i]:
            labels[i] = 0
    labels = np.array(labels)
    return labels

def line(X, k, b, gap=0.5):
    return k * X[:, 0] + b + gap

def split(X, labels):
    ratio = 1
    with_labels = labels != -1
    lower = labels == 0
    upper = labels == 1
    lower_count = sum(lower)
    upper_count = sum(upper)

    X_upper_train, y_upper_train = X[upper], labels[upper]
    X_lower_train, y_lower_train = X[lower], labels[lower]

    random_index = np.arange(int(upper_count))
    np.random.shuffle(random_index)
    X_upper_train, y_upper_train = X_upper_train[random_index], y_upper_train[random_index]
    upper_count = int(ratio * upper_count)
    X_upper_train, y_upper_train = X_upper_train[:upper_count, :], y_upper_train[:upper_count]
    random_index = np.arange(lower_count)
    np.random.shuffle(random_index)
    X_lower_train, y_lower_train = X_lower_train[random_index], y_lower_train[random_index]

    X_test, y_test = X[~with_labels], labels[~with_labels]
    X_train = np.r_[X_upper_train, X_lower_train]
    y_train = np.r_[y_upper_train, y_lower_train]
    # print(f'upper ~ lower: {upper_count}~{lower_count}')
    return (X_train, y_train), (X_test, y_test)

def fit(X, labels) -> Tuple[float, float]:
    model = LinearDiscriminantAnalysis()


    # Fit the classifier
    model.fit(X, labels)
    coefs = model.coef_[0]
    intercept = model.intercept_[0]
    k = -coefs[0]/coefs[1]
    b = -intercept/coefs[1]
    return k, b, model


# Function to plot decision boundary
def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.colorbar()
    plt.show()
    
    

def fit_params(init_k, savefig_filename=None):
    

    # 读取数据
    filename = "asteroid_dataframe.csv"   # the original data file name (csv format)
    X = read_data(filename)
    k, b, gap = init_k, -0.8 , 0.1
    clf = None
    plt.figure(figsize=(10, 8))
    for i in range(100):
        labels = classify(X, k=k, b=b, gap=gap)
        (X_train, y_train), (X_test, y_test) = split(X, labels)

        # 获取直线斜率
        k, b, clf = fit(X_train, y_train)
        print(f'Iteration: {i}, k: {k}, b: {b}')
        
        # plot
        plt.clf()
        y_pred = clf.predict(X)
        X1 = X[y_pred == 1]
        X2 = X[y_pred == 0]
        plt.scatter(10**X1[:, 0], 10**(-X1[:, 1]), s=1.0, label='Class I')
        plt.scatter(10**X2[:, 0], 10**(-X2[:, 1]), s=1.0, label = 'Class II')

        x_min, x_max = 0, 3
        x = 10**np.linspace(x_min, x_max, 100)
        plt.plot(x, 10**(-b) * x**(-k), color = 'black')
        plt.plot(x, 10**(-b-gap) * x**(-k))
        plt.plot(x, 10**(-b+gap) * x**(-k))
        # plt.plot(x, - (k * x + b), color = 'black')
        # plt.plot(x, - (k * x + b+gap))
        # plt.plot(x, - (k * x + b-gap))
        # plt.fill_between(x, - (k * x + b-gap), - (k * x + b+gap), color='grey', alpha=0.5)
        plt.gca().invert_yaxis()
        plt.legend(fontsize = 15)
        plt.xlabel('log(D) (km)', font = font1)
        plt.ylabel('log(P) (h)', font = font1)
        plt.title('Iteration %d:  $P_{\\rm h} = %.2f \\, D_{\\rm km}^{%.3f} $'%(i+1, 10**(-b),k), fontdict=font1)
        plt.xticks(fontproperties = 'Times New Roman', size = 20)
        plt.yticks(fontproperties = 'Times New Roman', size = 20)   
        plt.xscale('log')
        plt.yscale('log')
        # plt.xlim([0, 3])
        # plt.ylim([3.5, 0.2])
        plt.pause(0.1)
        # if i == 0 or i == 149:
        #     plt.savefig("gif/gap{}.pdf".format(i), bbox_inches='tight')
        # plt.savefig("gif/gap{}.png".format(i))
    
    # generate_gif(150)
 
if __name__ == '__main__':
    fit_params(init_k=-1.2)
    # plot_original_data()
    # imageio.imread("gif/gap{}.pdf".format(2))