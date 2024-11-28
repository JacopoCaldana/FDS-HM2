import numpy as np
from libs.libs_math import sigmoid
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class LogisticRegression:
    def __init__(self, num_features : int):
        self.parameters = np.random.normal(0, 0.01, num_features)
        
    def predict(self, x:np.array) -> np.array:
        
        ##############################
        #We take the dot product of input features x with the  model parameters and then we compute the sigmoid activation on it.
        preds = sigmoid(np.dot(x, self.parameters))
        ##############################
        return preds
    
    @staticmethod
    def likelihood(preds, y : np.array) -> np.array:
        
        ##############################
        #We calculate the binary cross-entropy loss with numerical stability adding a small constant.
        log_l = np.mean(y * np.log(preds + 1e-15) + (1 - y) * np.log(1 - preds + 1e-15))
        ##############################
        return log_l
    
    def update_theta(self, gradient: np.array, lr : float = 0.5):
        
        ##############################
        #We update model parameters using gradient descent.
        self.parameters += lr * gradient
        ##############################
        pass
        
    @staticmethod
    def compute_gradient(x : np.array, y: np.array, preds: np.array) -> np.array:
                    gradient: the gradient of the log likelihood.
       
        ##############################
        #Then we calculate the gradient for logistic regression parameters
        gradient = np.dot(x.T, (y - preds)) / x.shape[0]
        ##############################
        return gradient

