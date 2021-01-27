from matplotlib import pyplot as plt
import numpy as np
from numpy import random, zeros
import math


X = 2 * np.random.rand(100,1)
Y = 4 +3 * X+np.random.randn(100,1)

x_train = X
y_train = Y

X_b = np.c_[np.ones((len(x_train),1)),x_train]
best_result = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
print("Solution:")
print(best_result)



#gradient descent
#Based on: https://towardsdatascience.com/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc

def cost_f(param,x,y):

    n_samples = len(y)

    pred = x.dot(param)
    out = (1/2*n_samples)*np.sum(np.square(pred-y))
    return(out)
    
#the cost function expects a matrix of [1,X] because of the np.dot operation
X_b = np.c_[np.ones((len(x_train),1)),x_train]
print("Cost test:",cost_f([1,2],X_b,y_train))

def grad_desc(x,y,param,lr=0.01,iterations=100):
    
    n_samples = len(y)
    cost_hist = np.zeros(iterations)
    param_hist= np.zeros((iterations,2))


    print('Initial parameters:',param)
    
    for it in range(iterations):

        pred =np.dot(x,param)
        
        
        #updating params
        param = param - (1/n_samples)*lr*(x.T.dot((pred-y)))
        #print(param)
        
        #storing
        #print(param.T)
        param_hist[it,:] = param.T
        cost_hist[it]    = cost_f(param,x,y)

        #print('Processing:',it,'Current:',param)

    return param, cost_hist, param_hist



init_param = np.random.randn(2,1)
print('Initial parameters:',init_param)

#the cost function expects a matrix of [1,X] because of the np.dot operation
X_b = np.c_[np.ones((len(x_train),1)),x_train]
#running the gradient
param_f,cost_h,param_h = grad_desc(X_b,y_train,
                                   init_param,
                                   lr=0.01,
                                   iterations=500)
print("Final parameters:")
print(param_f)

