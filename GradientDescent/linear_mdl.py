from matplotlib import pyplot as plt
import numpy as np
from numpy import random, zeros
import math

#copy from: https://www.sohamkamani.com/blog/2018/01/28/linear-regression-with-python/
def generate_dataset_simple(beta, n, std_dev):
  # Generate x as an array of `n` samples which can take a value between 0 and 100
  x = np.random.uniform(0,100,size=n) 
  # Generate the random error of n samples, with a random value from a normal distribution, with a standard
  # deviation provided in the function argument
  e = np.random.randn(n) * std_dev
  # Calculate `y` according to the equation discussed
  y = x * beta + e
  return x,y

x, y = generate_dataset_simple(1, 50, 30)

# Take the first 40 samples to train, and the last 10 to test
x_train = x[:-10]
y_train = y[:-10]

x_test = x[-10:]
y_test = y[-10:]

#print(x_train)
#print(y_train)

#plt.scatter(x,y) #.savefig('data_in.png')
#plt.show()


### lazy functions

# Calculate root mean squared error stolen from https://machinelearningmastery.com/implement-machine-learning-algorithm-performance-metrics-scratch-python/
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return math.sqrt(mean_error)


def grid_calc(x,y,Mmat,Bmat):

  M_min,M_max = Mmat
  B_min,B_max = Bmat

  e_1 = rmse_metric(y,M_min*x+B_min)
  e_2 = rmse_metric(y,M_max*x+B_min)
  e_3 = rmse_metric(y,M_min*x+B_max)
  e_4 = rmse_metric(y,M_max*x+B_max)

  out_err = [e_1,e_2,e_3,e_4]
  out_all = [[M_min,B_min,e_1],
             [M_max,B_min,e_2],
             [M_min,B_max,e_3],
             [M_max,B_max,e_4]]

  #print(out_err)
  #print(min(out_err))
  #print(out_err.index(min(out_err)))
  best_pos = out_err.index(min(out_err))

  best_mdl = out_all[best_pos]
  print(best_mdl)
  
  return best_mdl

def find_best(x,y,Mmat,Bmat,niter):

  print("starting conditions")
  print(Mmat)
  print(Bmat)

  err = 0

  for i in range(1,(niter+1)):
    print('Iteration:',i)

    temp_mdl = grid_calc(x,y,Mmat,Bmat)
    #print(optim_mdl)

    
    #reducing the grid
    Mmat[0] = temp_mdl[0] - 1/2*temp_mdl[0]
    Mmat[1] = temp_mdl[0] + 1/2*temp_mdl[0]

    Bmat[0] = temp_mdl[1] - 1/2*temp_mdl[1]
    Bmat[1] = temp_mdl[1] + 1/2*temp_mdl[1]

    print(Mmat)
    print(Bmat)

    #print(Bmat)
    if (i == 1 or temp_mdl[2] < err):
      err=temp_mdl[2] 
      optim_mdl = temp_mdl
  
    

  print(temp_mdl)
  print(optim_mdl)

  #return optim_mdl



#out = grid_calc(x_train,y_train,[1,2],[1,2])
out = find_best(x_train,y_train,[0,50],[0,50],10)



print("final",out)
#print(out[0][2])
#print(rmse_metric(y_train,y_train))
#print(type(rmse_metric(y_train,y_train)))
#print(math.sqrt(0))  


