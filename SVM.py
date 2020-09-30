#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, constraint, inputs, targets, arg):
        self.N = inputs.shape[0]                        # Size of input vector

         # For eq 4 in objective()
        self.t = targets.reshape(self.N, 1)             # -1, 1 for datapoints
        self.x = inputs                                 # input vector    
        self.C = constraint                             # 

        # Datapoints
        self.zerofunlist = list()
        self.kern = arg

        # Python list comprehension to make a list of items
        self.P = np.array([[ti*tj*self.kernel(xi, xj) for tj, xj in zip(self.t, self.x)] for ti, xi in zip(self.t, self.x)])             # Matrix in objective()                                
        self.B = np.asarray([(0, self.C) for b in range(self.N)])    
        self.alpha = np.zeros(self.N).reshape(self.N, 1)
        
    # Implements equations in section 3.3
    def kernel(self, x, y, r=1, p=2, sigma=1):
        # Linear kernel K(x,y) = x' * y
        # Polynomial kernel K(x,y) = (x' * y + r)^p
        # Radial Basis Functions kernel K(x,y) = e^-(||x-y||^2 / (2*sigma^2))
        if self.kern == "linear":
            ret = np.dot(x,y)
        elif self.kern == "poly":
            ret = np.power((np.dot(x, y) + r), p)
        elif self.kern == "rbf":
            ret = np.exp( -(np.linalg.norm(x-y)**2) / (2*sigma^2) )
        return ret

    def start(self):
        return np.zeros(self.N)

    # Implements equation 4
    def objective(self, alfa):
        # Alpha values matrix
        alfa_m = np.dot(alfa, alfa.T)
        # Objective function
        sum_tot = np.dot(alfa_m, self.P)
        # Return sum
        return np.sum(sum_tot - alfa)

    # Implements equation 10
    # Takes vector in as a parameter
    def zerofun(self, alfa):
        product = alfa*self.t

        """if np.sum(product)==0:
            return np.sum(product)
        else:
            print('Equality constraint not fulfilled')"""
        return np.sum(product)

    # Extracts non zero values and adds indices and values into a dictionary
    # Extracts support vectors referred to as s in lab doc
    def nonZeroExtract(self, alfa):
        ind = np.where(alfa > 1.0e-5)
    
        dic = {'ind': ind[0],
                'targets': self.t[ind[0]],
                'inputs': self.x[ind[0]],
                'alpha': alfa[ind[0]]}

        self.zerofunlist.append(dic)
        
        return self.zerofunlist


    # Implements equation 7
    def calculate_b(self, alfa, s_v, ts_v):
        # sum(alfa_i*t_i*K(s, x) - t_s)
        # alfa_i*t_i

        b_list = [[a_i*t_i*self.kernel(s_j, x_i) - ts_j for a_i, t_i, x_i in zip(self.alpha, self.t, self.x)] for s_j, ts_j in zip(s_v, ts_v)]
        b = np.sum(b_list)
        return b

    def minimize(self):
        # Constraints
        Xc = {'type':'eq', 'fun':self.zerofun}
        # Call to scipy minimize
        ret = minimize(self.objective, np.zeros(self.N), bounds=self.B, constraints=Xc)
        if ret['success'] == True: 
            #print("Minimize success") 
            self.alpha = ret['x']
            #print(self.alpha)
        else:
            print("Minimize did not find a solution")
            return

        return
        
    # Implements equation 6
    def indicator(self, zerofunlist, b, s):
        # ind(s) = sum(alfa_i*t_i*K(s, x_i) - b)
        ind_s = [[alfa_i*t_i*self.kernel(s, x_i) for alfa_i, t_i, x_i in zip(self.alpha, self.t, self.x)] for s in self.zerofunlist[0]['inputs']] - self.calculate_b
        return ind_s

def plot_func(class_a, class_b):
    for p in class_a:
        plt.plot(p[0],p[1],'b')

    for p in class_b:
        plt.plot(p[0],p[1],'r')
    
    plt.axis('equal') # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen
    plt.show()
    
def plot_boundary():
    xgrid=np.linspace(-5, 5)
    ygrid=np.linspace(-4, 4)
    grid=np.array([[indicator(x, y) for x in xgrid ] for y in ygrid])
    plt . contour ( xgrid , ygrid , grid , (-1, 0.0, 1.0),
    colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    
    
def main():
    # Data generation
    np.random.seed(100)
    class_a = np.concatenate((np.random.randn(10, 2)*0.2 + [1.5, 0.5], 
                                np.random.randn(10, 2)*0.2 + [-1.5, 0.5]))
    class_b = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
    
    inputs = np.concatenate((class_a, class_b))
    targets = np.concatenate((np.ones(class_a.shape[0]), -np.ones(class_b.shape[0])))

    N = inputs.shape[0]
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    # Each datapoint in inputs has an x and y value
    x = inputs[:, 0]
    y = inputs[:, 1]

    s = x
    t_s = targets
    N = targets.shape[0]
    alfa = np.zeros(x.shape[0]).reshape(x.shape[0], 1)

    svm = SVM(0.5, inputs, targets, "linear")

    #temp1 = alfa*t_s
    #print(inputs[0:2, 0:2])
    #print(np.dot(inputs[0:2, 0:2].T, inputs[0:2, 0:2]))
    
    n = svm.nonZeroExtract(alfa)
    print(n)
    m = svm.zerofun(alfa)
    print(m)
    #print(d)
    d = svm.calculate_b(alfa, s, t_s)
    svm.nonZeroExtract(np.arange(40))
    #svm.nonZeroExtract(np.arange(40))
    #svm.nonZeroExtract(np.arange(40))
    #print(len(svm.zerofunlist))
    

 


if __name__ == "__main__":
    main()