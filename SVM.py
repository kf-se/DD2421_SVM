#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, constraint, inputs, targets, arg):
        # Global parameters
        self.N = inputs.shape[0]                        # Size of input vector

         # For eq 4 in objective()
        self.t = targets.reshape(self.N, 1)             # -1, 1 for datapoints
        self.x = inputs                                 # input vector    
        self.C = constraint                             # 

        # Datapoints
        self.zerofunlist = list()
        self.kern = arg

        # Python list comprehension to make a list of items
        self.P = np.array([[ti*tj*self.kernel(xi, xj) for tj, xj in zip(self.t, self.x)] for ti, xi in zip(self.t, self.x)])    # Should be NxN-matrix
        # print(np.shape(self.P))                         
      
        # Matrix in objective()                                
        self.B = np.asarray([(0, self.C) for b in range(self.N)])    
        self.alpha = np.zeros(self.N).reshape(self.N, 1)
        
        # start vector
        self.start = np.zeros(self.N)

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


    # def start(self):
        # return np.zeros(self.N)


    # Implements equation 4
    # Used in minimize
    # INPUT: vector
    # OUTPUT: scalar
    def objective(self, alfa):
        # Alpha values matrix

        alfa_m = np.dot(alfa, alfa.T)                
        #print('alfa_m dim=', np.shape(alfa_m)) 
        #alfa_m = [alfa_i*alfa_j for alfa_i, alfa_j in zip(alfa, alfa.T)]
        # print('alfa_m dim=', np.shape(alfa_m)) 

        # Objective function
        sum_tot = 0.5*np.dot(alfa_m, self.P)

        #print('sum_tot dim=', np.shape(sum_tot)) 
        eq4 = sum_tot - np.sum(alfa)
        # print(eq4)
        # Return sum
        return eq4

    # Implements equation 10
    # Takes vector in as a parameter
    # INPUT: vector (data before separation)
    # OUTPUT: scalar, value that should be constrained to zero
    def zerofun(self, alfa):
        product = alfa*self.t
        return product
        #if np.sum(product)==0:
        #    return np.sum(product)
        #else:
        #    print('Equality constraint not fulfilled')

    # Extracts non zero values and adds indices and values into a dictionary
    # Extracts support vectors referred to as s in lab doc
    # INPUT: vector, result (data after separation)
    # OUTPUT: vector, error (data being missclassified)
    def nonZeroExtract(self, alfa):
        ind = np.where(alfa > 1.0e-5)
    
        dic = {'ind': ind[0],
                'targets': self.t[ind[0]],
                'inputs': self.x[ind[0]],
                'alpha': alfa[ind[0]]}

        self.zerofunlist.append(dic)
        
        return self.zerofunlist


    # Implements equation 6
    # 
    # INPUT:
    # OUTPUT: classification (ind < -1 (class -1) or ind > 1 (class 1))
    def indicator(self, x, y, i):
        # ind(s) = sum(alfa_i*t_i*K(s, x_i) - b)
        ind_s = [alfa_i*t_i*self.kernel([x, y], x_i) for alfa_i, t_i, x_i in zip(self.zerofunlist[i]['alpha'], self.zerofunlist[i]['targets'], self.zerofunlist[i]['inputs'])] - self.calculate_b(i)
        return ind_s

        
    #def plot_boundary(self, i):
     #   xgrid=np.linspace(-50, 50)
     #   ygrid=np.linspace(-40, 40)
     #   grid=np.array([[self.indicator(x,y,) for x in xgrid ] for y in ygrid])
     #   plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

    # Implements equation 7
    # Calculate b used in indicator
    # INPUT: vector 
    # OUTPUT: scalar, b
    def calculate_b(self, i):
        # sum(alfa_i*t_i*K(s, x) - t_s)
        # alfa_i*t_i
        b_list = [[a_i*t_i*self.kernel(s_j, x_i) - ts_j for a_i, t_i, x_i in zip(self.alpha, self.t, self.x)] for s_j, ts_j in zip(self.zerofunlist[i]['inputs'], self.zerofunlist[i]['targets'])]
        b = np.sum(b_list)
        return b


    # minimizes objective
    # INPUT:
    # OUTPUT: vector  
    def minimize(self, alfa):
        # Constraints
        XC = {'type':'eq', 'fun':self.zerofun}
        objective = self.objective(alfa)
        # Call to scipy minimize
        ret = minimize(objective, np.zeros(self.N), bounds=self.B, constraints=XC)
        if ret['success'] == True: 
            #print("Minimize success") 
            self.alpha = ret['x']
            #print(self.alpha)
            return (self.alpha)
        else:
            print("Minimize did not find a solution")
            return

        return
        

def plot_func(class_a, class_b, svm):
    for p in class_a:
        plt.plot(p[0],p[1],'b.')

    for p in class_b:
        plt.plot(p[0],p[1],'r.')

    for p in svm.zerofunlist[0]['inputs']:
        plt.plot(p[0], p[1], 'g+')
    
    plt.axis('equal') # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen
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
    #print(x)
    #print(y)

    #s = x
    #t_s = targets
    #N = targets.shape[0]
    alfa = np.zeros(x.shape[0]).reshape(x.shape[0], 1)
    alfa1 = np.arange(N).reshape(N,1)

    svm = SVM(0.5, inputs, targets, "linear")
    C = 10
    B=[(0, C) for b in range(N)]
    start=np.zeros(N)
    XC = {'type':'eq', 'fun':svm.zerofun}
    alpha = svm.minimize(alfa)

    #svm.minimize()
    #svm.objective(alfa)
    #print("alfa:", alfa1)
    #print("targets:", targets, "sum of targets", np.sum(targets))
    #print("inputs:", inputs)

    # Test zerofun
    #print("zerofun:", svm.zerofun(alfa))

    # Test nonZeroExtract
    #print("nonZeroExtract:", svm.nonZeroExtract(np.arange(10)))
    #print("Nonzerolist:", svm.zerofunlist)

    # Test Kernel
    #print("Kernel:", svm.kernel(inputs[0, :], inputs[1, :]),"of point a:", inputs[0, :], "and b:", inputs[1, :])

    # Test objective
    #print("Objective:", svm.objective(alfa1))

    # Test indicator
    #indicator = svm.indicator(1, 2, 0)
    #print("Indicator", indicator)
    #print(indicator.shape)
    #print(np.sum(indicator))

    # Test calculate_b
    #print("calculate_b is done")

    # Test minimize 
    #svm.minimize()

    # plot_func(class_a, class_b, svm)
    # svm.plot_boundary(2)

if __name__ == "__main__":
    main()