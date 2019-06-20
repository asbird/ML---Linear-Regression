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
def h_x(theta, x):
    """
        Hypothesis function
    """
    return x.dot(theta.values)

def cost_deriv(x, y, theta, feature):
    """
        Derivative of cost function
    """
    hx = h_x(theta, x)
    sus = -y.sub(hx,0) #minus at the beggining because eq is hh-y => == -(-hh+y)
    sum = 0
    for index, row in x.iterrows():
        print(index)
        h_xi = h_x(theta, row) # ok
        yi =  y.at[index,'price']
        sum += (h_xi - yi)*X.at[index,feature]
    pdb.set_trace()

# def gradient(x, y, theta, iterations, alpha):
#     features = list(my_dataframe)
#     numOfTheta = theta.shape[0]
#     tempTheta = np.zeros(numOfTheta)
#     for i in range(iterations):
#         for i in numOfTheta:






def FeatureScalling(X):
    print("Starting feature scalling...\nPlease Wait.")
    printProgressBar(0, len(X.index), prefix = 'Feature scalling progress:', suffix = 'Complete', length = 50)
    for index, row in X.iterrows():
        for feature in list(X):
            _prev=X.at[index,feature]
            _current = (X.at[index,feature] - X[feature].mean())/(X[feature].max() - X[feature].min())
            X.at[index,feature] = _current
            printProgressBar(index, len(X.index), prefix = 'Feature scalling progress:', suffix = 'Complete', length = 50)
            # DEBUGGING
            # print(str(index) +': Feature:' + str(feature) +
            # ' Prev: '+str(_prev)+' Current: '+str(_current))


df = pd.read_csv('house_data.csv', sep=',')

# features = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']
features = ['sqft_living', 'grade']

X = df[features]
y = df[['price']]

theta = np.array([2,2])

# FeatureScalling(X)
# cost_deriv(X, y, theta)
