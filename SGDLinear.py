import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import pdb
from numpy import genfromtxt

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def FeatureScalling(X):
    print("Starting feature scalling...\nPlease Wait.")
    printProgressBar(0, len(X.index), prefix = 'Feature scalling progress:', suffix = 'Complete', length = 50)
    for index, row in X.iterrows():
        for feature in list(X):
            if feature == 'ones':
                continue
            _prev=X.at[index,feature]
            _current = (X.at[index,feature] - X[feature].mean())/(X[feature].max() - X[feature].min())
            X.at[index,feature] = float(_current)
            printProgressBar(index, len(X.index), prefix = 'Feature scalling progress:', suffix = 'Complete', length = 50)
            # DEBUGGING
            # print(str(index) +': Feature:' + str(feature) +
            # ' Prev: '+str(_prev)+' Current: '+str(X.at[index,feature]))

def h_x(Theta, x):
    """
        Hypothesis function
    """
    return x.dot(Theta.values)

def cost_fct(x, y, Theta):
    bs=len(x.index)
    hx = h_x(Theta, x)
    hxSuby = hx.sub(y.values)
    suM = hxSuby.sum(axis = 0, skipna = True)
    x = pow(suM, 2)
    return x/bs

def BGD(theta, data_x, data_y, alpha, iterations, bs):
    # pdb.set_trace()
    batch = len(data_x.index) / bs
    for i in range(0,iterations+1):
        b=1
        for j in range(0, len(data_x.index), bs):
            # pdb.set_trace()
            if j>0:
                theta = NewTheta
            hxSuby = h_x(theta, data_x[j:j+bs]).sub(data_y[j:j+bs].values)
            hxSubyMulX = data_x[j:j+bs].mul(hxSuby.values,0)
            dL = alpha*(2/bs)*hxSubyMulX.sum(axis = 0, skipna = True)
            NewTheta = theta.sub(dL.values, 0)
            print('Epoch: '+str(i)+'/'+str(iterations)+' Batch: '+str(b)+'/'+str(batch)
            +' Loss: '+str(cost_fct(data_x[j:j+bs], data_y[j:j+bs], NewTheta)))
            b=b+1
    return NewTheta




df = pd.read_csv('house_data.csv', sep=',')

# features = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']
features = ['sqft_living', 'sqft_living', 'grade', 'sqft_above', 'bathrooms']

X = df[features]
y = df[['price']]
X.columns=['ones', 'sqft_living', 'grade', 'sqft_above', 'bathrooms']
# X=X[:5000]
# y=y[:5000]
X['ones'] = 1
X=X.astype(np.float64)

theta = pd.DataFrame(np.random.randint(-1000,1000,size=(5,1)), columns=['theta'])
# cost_fct(X, y, theta)
FeatureScalling(X)
theta = BGD(theta, X, y, 0.001, 3,10)
# X = pd.concat([ones, X], axis=1, sort=False)
# Slicing data for the purpose of creating algorithms and checking
# if they work correctly



asd1 = h_x(theta, X.iloc[[1]])
asd3 = h_x(theta, X.iloc[[3]])
asd5 = h_x(theta, X.iloc[[5]])
pdb.set_trace()

# cost_deriv(X, y, theta)
