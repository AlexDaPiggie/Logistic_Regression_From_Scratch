import numpy as np 
def sigmoid(z):
    z = np.clip (z, -500, 500) # Cap z to prevent overflow
    result = 1.0 / (1.0 + np.exp (-z)) 
    return result

def calculate_gradient(theta, X, y, lambda_reg = 0):
    '''
    Calculate the gradient for gradient descent
    '''
    m = y.size
    grad = (X.T @ (sigmoid (X @ theta) - y)) / m
    grad[1:] += (lambda_reg / m) * theta[1:]
    return grad

class LogisiticRegression:
    def __init__(self, num_iter = 1000, alpha = 0.01, tol = 1e-8, threshold = 0.5, lambda_reg = 0):
        self.theta = None
        self.num_iter = num_iter
        self.alpha = alpha
        self.tol = tol
        self.threshold = threshold
        self.lambda_reg = lambda_reg

    def fit(self, X, y): #Optimizing paramters using 'gradient descent'
        X_b = np.c_[np.ones((X.shape[0], 1)), X] # X_b is X with intercept
        self.theta = np.zeros (X_b.shape[1]) # Theta is also with intercept, and is initialized as 0

        for _ in range (self.num_iter):
            grad = calculate_gradient(self.theta, X_b, y, self.lambda_reg)
            self.theta -= self.alpha * grad

            if np.linalg.norm (grad) <= self.tol: 
                break

        return self.theta

    def predict_proba(self, X): 
        '''
        return the probability that the ouput is 1
        '''
        X_b = np.c_[np.ones((X.shape[0], 1)), X] # X_b is X with intercept
        return sigmoid (X_b @ self.theta)

    def predict(self, X): 
        '''
        Compare the probability with threshold to dictate final result (0 or 1)
        '''
        return (self.predict_proba(X) >= self.threshold).astype(int) # Output is either 0 or 1