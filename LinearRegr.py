import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


df = pd.read_csv('house_data.csv', sep=',')
corr = df.corr()

ax = sns.heatmap(corr, cmap='coolwarm_r',center=0)
plt.show()

params = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']
# Iteration through Data frame
# for index, row in df.iterrows():
#     print(row['id'])


def h_x(params, x):
    print("xd")
