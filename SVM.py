import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, t, size_d):
        self.N = size_d
        self.t = t          # For eq 4 in objective()
        self.K = K          # For eq 4 in objective()
        self.P = 1          # Matrix to be used in objective()
        
    # Implements equations in section 3.3
    def kernel(self, arg):
        # Linear kernel K(x,y) = x' * y
        # Polynomial kernel K(x,y) = (x' * y + r)^p
        # Radial Basis Functions kernel K(x,y) = e^-(||x-y||^2 / (2*sigma^2))
        1

    # Implements equation 4
    def objective(self, alfa):
        1

    # Implements equation 10
    def zerofun(self, alfa):
        1

    def nonZeroExtract(self, alfa):
        1

    def minimize(self):
        # Call to scipy minimize
        # ret = minimize(objective, start, bounds=B, constraints=Xc)
        # alpha = ret['x']
        # where B = [(0, C) for b in range(N)]
        1
    
    # Implements equation 6
    def indicator(self):
        1


def main():
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
    print(targets)
    print(inputs)




if __name__ == "__main__":
    main()
