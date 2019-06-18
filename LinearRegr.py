import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import pdb




# corr = df.corr()
#
# ax = sns.heatmap(corr, cmap='coolwarm_r',center=0)
# plt.show()
#
# Iteration through Data frame
# for index, row in df.iterrows():
#     print(row['id'])
def h_x(theta, x):
    theta = theta.transpose()
    return np.matmul(theta,x)

def FeatureScalling(X):
    for index, row in X.iterrows():
        for feature in list(X):
            _prev = X.at[index,feature]
            _mean = X[feature].mean()
            _max = X[feature].max()
            _min = X[feature].min()
            _current = (_prev - _mean) / (_max - _min)
            X.at[index,feature] = _current

            # DEBUGGING
            # print(str(index) +': Feature:' + str(feature) +
            # ' Prev: '+str(_prev)+' Current: '+str(_current))

df = pd.read_csv('house_data.csv', sep=',')
features = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']

X = df[features]
y = df[['price']]

FeatureScalling(X)
