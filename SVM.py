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
        self.B = np.array([(0, self.C) for b in range(self.N)]) 

        # Datapoints
        self.zerofunlist = dict()
        self.kern = arg

        # Python list comprehension to make a list of items
        self.alpha = np.zeros(self.N).reshape(self.N, 1)
        self.P = np.array([[ti*tj*self.kernel(xi, xj) for tj, xj in zip(self.t, self.x)] for ti, xi in zip(self.t, self.x)]).reshape(self.N, self.N)             # Matrix in objective()    
        self.P2 = []
        for i in range(self.N):
            A = []
            for j in range(self.N):
                k = self.kernel(inputs[i], inputs[j])
                A.append(targets[i]*targets[j]*k)
            self.P2.append(np.array(A))
        

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
    # Used in minimize
    # INPUT: vector
    # OUTPUT: scalar
    def objective(self, alfa):
        # Alpha values matrix
        alfa_m = np.dot(alfa, alfa.T)
        # Objective function
        sum_tot = np.dot(alfa_m, self.P)
        ret = np.sum(0.5*sum_tot) - np.sum(alfa)

        p = 0
        for i in range(self.N):
            for j in range(self.N):
                p += alfa[i]*alfa[j]*self.P[i][j]
        ret1 = 0.5*p - np.sum(alfa)
        
        ret2 = 0.5*np.dot(alfa, np.dot(self.P, alfa)) - np.sum(alfa)
        # Return sum
        return ret1


    # Implements equation 10
    # Takes vector in as a parameter
    # INPUT: vector (data before separation)
    # OUTPUT: scalar, value that should be constrained to zero
    def zerofun(self, alfa):
        return np.dot(alfa, self.t)


    # Extracts non zero values and adds indices and values into a dictionary
    # Extracts support vectors referred to as s in lab doc
    # INPUT: vector, result (data after separation)
    # OUTPUT: vector, error (data being missclassified)
    def nonZeroExtract(self):
        ind = np.where(self.alpha > 1.0e-5)
    
        dic = {'ind': ind[0],
                'targets': self.t[ind[0]],
                'inputs': self.x[ind[0]],
                'alpha': self.alpha[ind[0]]}

        self.zerofunlist = dic
        return self.zerofunlist


    # Implements equation 6
    # 
    # INPUT:
    # OUTPUT: classification (ind < -1 (class -1) or ind > 1 (class 1))
    def indicator(self, x, y):
        # ind(s) = sum(alfa_i*t_i*K(s, x_i) - b)
        ind_s = np.array([alfa_i*t_i*self.kernel([x, y], x_i) - self.calculate_b() for alfa_i, t_i, x_i in zip(self.zerofunlist['alpha'], self.zerofunlist['targets'], self.zerofunlist['inputs'])]) 
        #print(ind_s.shape)
        return np.sum(ind_s)


    # Implements equation 7
    # Calculate b used in indicator
    # INPUT: vector 
    # OUTPUT: scalar, b
    def calculate_b(self):
        # sum(alfa_i*t_i*K(s, x) - t_s)
        # alfa_i*t_i
        b_list = np.array([[a_i*t_i*self.kernel(s_j, x_i) - ts_j for a_i, t_i, x_i in zip(self.alpha, self.t, self.x)] for s_j, ts_j in zip(self.zerofunlist['inputs'], self.zerofunlist['targets'])])
        b = np.sum(b_list)
        return b


    # minimizes objective
    # INPUT:
    # OUTPUT: vector  
    def minimize(self):
        # Constraints
        XC = {'type':'eq', 'fun':self.zerofun}
        # Call to scipy minimize
        ret = minimize(self.objective, np.zeros(self.N), bounds=self.B, constraints=XC)  
        if ret['success'] == True: 
            print("Minimize success") 
            self.alpha = ret['x']
        else:
            print("Minimize did not find a solution")
            
        

def plot_func(class_a, class_b, svm):
    for p in class_a:
        plt.plot(p[0],p[1],'b.')

    for p in class_b:
        plt.plot(p[0],p[1],'r.')

    for p in svm.zerofunlist['inputs']:
        plt.plot(p[0], p[1], 'g+')
    
    plt.axis('equal') # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen
    
    
    #if(pltshow = True):
    xgrid=np.linspace(-5, 5)
    ygrid=np.linspace(-4, 4)
    print([x for x in xgrid ])
    #print("grid", len([[svm.indicator(x, y) for x in xgrid ] for y in ygrid]))
    grid=np.array([[svm.indicator(x, y) for x in xgrid ] for y in ygrid])
    #print(grid)
    #grid=np.array([[svm.indicator(x, y), svm.zerofunlist['alpha'], svm.zerofunlist['inputs'], svm.zerofunlist['targets'] for x in xgrid ] for y in ygrid])
    plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    
    plt.show()
    
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
    alfa1 = np.arange(N).reshape(N,1)

    svm2 = SVM(None, inputs, targets, "linear")
    ret = svm2.minimize()
    svm2.nonZeroExtract()
    #print(svm2.zerofunlist['targets'])
    #print(svm2.zerofunlist)
    #svm2.calculate_b()
    #svm2.nonzeroextract()
    plot_func(class_a, class_b, svm2)
    #plot_boundary(svm2)


if __name__ == "__main__":
    main()