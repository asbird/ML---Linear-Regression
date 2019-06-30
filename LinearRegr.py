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

# Math equations
def h_x(Theta, x):
    """
        Hypothesis function
    """
    return x.dot(Theta.values)

def h_x2(Theta, x):
    #Setting second row to 1 because x2 doesn't need to be multiplied
    #according to x0=1, x1, x2^2, x2^3  etc.
    x = x_power(x)
    return x.dot(Theta.values)

def x_power(x):
    xPow = x.copy()
    xPow[x.columns[1]]=1.0
    p=2
    for idea in range(len(x.columns)-2):
        x=x*xPow
        xPow[x.columns[p]]=1.0
        p=p+1
    return x

def cost_fct(x, y, Theta):
    m=len(x.index)
    hx = h_x2(Theta, x)
    hxSuby = hx.sub(y.values)
    suM = hxSuby.sum(axis = 0, skipna = True)
    x = pow(suM, 2)
    return x/(2*m)

def cost_deriv(x, y, Theta):
    """
        Derivative of cost function 0o*x0 + 01*x1 + 02x2+ 03x3 etc.
    """
    m=len(X.index)
    hx = h_x(Theta, x)
    hxSuby = hx.sub(y.values)
    hxSubyMulX = x.mul(hxSuby.values,0)
    suM = hxSubyMulX.sum(axis = 0, skipna = True)
    return suM.mul(1/m)

def cost_deriv2(x, y, Theta):
    """
        Derivative of cost function 0o*x0 + 01*x1 + 02x2^1+ 03x3^2 etc.
    """
    m=len(X.index)
    hx = h_x2(Theta, x)
    hxSuby = hx.sub(y.values)
    x=x_power(x)
    hxSubyMulX = x.mul(hxSuby.values,0)
    suM = hxSubyMulX.sum(axis = 0, skipna = True)
    return suM.mul(1/m)

def calc_new_theta(x, y, Theta, alpha):
    features = list(x)
    tempTheta = Theta
    cd = cost_deriv2(x, y, Theta).mul(alpha)
    for i in range(len(features)):
        tempTheta.at[i,'theta'] = theta.at[i,'theta']-cd.at[features[i]]
    return tempTheta

def gradient_descent(x, y, Theta, alpha, iterations):
    errors = []
    costs = []

    actual_cost=0
    print('Starting Gradient descent algorithm...\nPlease Wait')
    # printProgressBar(0, iterations, prefix = 'Gradient descent progress:', suffix = 'Complete', length = 50)
    for i in range(iterations):
        Theta=calc_new_theta(x, y, Theta, alpha)
        prev_cost = actual_cost
        actual_cost = cost_fct(x, y, Theta)
        # printProgressBar(i, iterations, prefix = 'Gradient descent progress:', suffix = 'Complete', length = 50)
        if int(actual_cost)<10:
            break
        if i>0:
            Error = float(prev_cost)-float(actual_cost)
            errors.append(Error)
            costs.append(actual_cost)
            print("Cost: "+str(float(actual_cost)))
            print("Error:"+str(float(Error)))

    print('Gradient descent algorithm has been succesfully completed')

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(costs)), costs)
    plt.title('Iterations')
    plt.ylabel('Cost fct')

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(errors)), errors)
    plt.xlabel('Iterations')
    plt.ylabel('Errors')

    plt.show()

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


df = pd.read_csv('house_data.csv', sep=',')

# features = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']
features = ['sqft_living', 'sqft_living', 'grade', 'sqft_above', 'bathrooms']

X = df[features]
y = df[['price']]
X.columns=['ones', 'sqft_living','grade', 'sqft_above', 'bathrooms' ]
X=X[:1000]
y=y[:1000]
X['ones'] = 1

# X = pd.concat([ones, X], axis=1, sort=False)
# Slicing data for the purpose of creating algorithms and checking
# if they work correctly
X=X.astype(np.float64)
theta = pd.DataFrame(np.random.randint(-1000,1000,size=(5,1)), columns=['theta'])
# h_x2(X, y, theta)

FeatureScalling(X)


gradient_descent(X, y, theta, 0.4, 5000)
asd1 = h_x(theta, X.iloc[[1]])
asd3 = h_x(theta, X.iloc[[3]])
asd50 = h_x(theta, X.iloc[[50]])
pdb.set_trace()
