import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, constraint, inputs, targets, arg):
         # For eq 4 in objective()
        self.t = targets                                # -1, 1 for datapoints
        self.x = inputs                                 # input vector    
        self.C = constraint                             # 

        # Datapoints
        self.N = inputs.shape[0]                        # Size of input vector
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
            ret = np.dot(x.T, y)
        elif self.kern == "poly":
            ret = np.power((np.dot(x.T, y) + r), p)
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
        return np.sum(sum_tot)

    # Implements equation 10
    # Takes vector in as a parameter
    def zerofun(self, alfa):
        product = np.dot(alfa.T, self.t)

        """if np.sum(product)==0:
            return np.sum(product)
        else:
            print('Equality constraint not fulfilled')"""
        return np.sum(product)

    # Extracts non zero values and adds indices and values into a dictionary
    def nonZeroExtract(self, alfa):
        ind = np.where(alfa > 1.0e-5)
        dic = {'ind': ind[0],
                'targets': self.t[ind[0]],
                'inputs': self.x[ind[0]]}
        self.zerofunlist.append(dic)

    def calculate_b(self, alfa, s):
        # sum(alfa_i*t_i*K(s, x) - t_s)
        1

    def minimize(self):
        # Constraints
        Xc = {'type':'eq', 'fun':self.zerofun}
        # Call to scipy minimize
        ret = minimize(self.objective, np.zeros(self.N), bounds=self.B, constraints=Xc)
        if ret['success'] == True: 
            #print("Minimize success") 
            self.alpha = ret['x']
        else:
            print("Minimize did not find a solution")
            return
       
        return
        
    # Implements equation 6
    def indicator(self):
        1

def plot_func(class_a, class_b):
    for p in class_a:
        plt.plot(p[0],p[1],'b')

    for p in class_b:
        plt.plot(p[0],p[1],'r')
    
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

    x = inputs[:, 0]
    y = inputs[:, 1]
    alfa = np.zeros(x.shape[0]).reshape(x.shape[0], 1)

    svm = SVM(0.5, inputs, targets, "linear")
    for i in range(10):
        svm.minimize()
        #print("zerofun: ", svm.zerofun(alfa))
    #print("alfa post: ", svm.alpha)
    svm.nonZeroExtract(targets)
    # ko = svm.kernel(x, y)
    # o = svm.objective(alfa)


if __name__ == "__main__":
    main()
