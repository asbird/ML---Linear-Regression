import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import pdb
from numpy import genfromtxt




# corr = df.corr()
#
# ax = sns.heatmap(corr, cmap='coolwarm_r',center=0)
# plt.show()
#
# Iteration through Data frame
# for index, row in df.iterrows():
#     print(row['id'])

# Print iterations progress
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


# Math equations
def h_x(Theta, x):
    """
        Hypothesis function
    """
    return x.dot(Theta.values)

def cost_fct(x, y, Theta):
    m=len(x.index)
    hx = h_x(Theta, x)
    hxSuby = hx.sub(y.values)
    suM = hxSuby.sum(axis = 0, skipna = True)
    x = pow(suM, 2)
    return x/(2*m)

def cost_deriv(x, y, Theta):
    """
        Derivative of cost function
    """
    m=len(X.index)
    hx = h_x(Theta, x)
    hxSuby = hx.sub(y.values)
    hxSubyMulX = x.mul(hxSuby.values,0)
    suM = hxSubyMulX.sum(axis = 0, skipna = True)
    return suM.mul(1/m)

def calc_new_theta(x, y, Theta, alpha):
    features = list(x)
    cd = cost_deriv(x, y, Theta).mul(alpha)
    tempTheta = Theta
    for i in range(len(features)):
        tempTheta.at[i,'theta'] = theta.at[i,'theta']-cd.at[features[i]]
    return tempTheta

def gradient_descent(x, y, Theta, alpha, iterations):
    costs = []
    for i in range(iterations):
        tempTheta=calc_new_theta(x, y, Theta, alpha)
        Theta=tempTheta
        actual_cost = cost_fct(x, y, Theta)
        print(actual_cost)
        costs.append(actual_cost)
    plt.plot(costs,np.arange(len(costs)))
    plt.show()




def FeatureScalling(X):
    # print("Starting feature scalling...\nPlease Wait.")
    printProgressBar(0, len(X.index), prefix = 'Feature scalling progress:', suffix = 'Complete', length = 50)
    for index, row in X.iterrows():
        for feature in list(X):
            _prev=X.at[index,feature]
            _current = (X.at[index,feature] - X[feature].mean())/(X[feature].max() - X[feature].min())
            X.at[index,feature] = float(_current)
            # printProgressBar(index, len(X.index), prefix = 'Feature scalling progress:', suffix = 'Complete', length = 50)
            # DEBUGGING
            # print(str(index) +': Feature:' + str(feature) +
            # ' Prev: '+str(_prev)+' Current: '+str(X.at[index,feature]))


df = pd.read_csv('house_data.csv', sep=',')

# features = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']
features = ['sqft_living', 'grade']

X = df[features]
y = df[['price']]

# Slicing data for the purpose of creating algorithms and checking
# if they work correctly
X = X[:1000]
y = y[:1000]
X=X.astype(np.float64)

FeatureScalling(X)
theta = pd.DataFrame({'theta':[2,2]})
# cost_deriv(X, y, theta)
# gradient_descent(X, y, theta, 20, 10)
gradient_descent(X, y, theta, 0.01, 1000)

# cost_deriv(X, y, theta)
