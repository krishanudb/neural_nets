import numpy as np

class FCN1():
    
    def __init__(self, start, layers, end):
        
        self.starting_dim = start
        self.hidden_dim = layers
        self.output_dim = end
        self.dimensions = [start,layers,end]
                        
        self.parameters = self.initialize_parameters()
        self.temps = {}
        self.grads = {}
        
    def initialize_parameters(self):
        dims = self.dimensions
        
        parameters = {}
        
        parameters["W1"] = np.random.randn(dims[1], dims[0]) * 0.01
        parameters["b1"] = np.zeros((dims[1], 1))

        parameters["W2"] = np.random.randn(dims[2], dims[1]) * 0.01
        parameters["b2"] = np.zeros((dims[2], 1))
        
        return parameters
                
    def train(self, X, Y, iterations = 1000, learning_rate = 0.0001):
        for i in range(iterations):
            self.one_run(X, Y)
    
    def predict(self, X):
        return self.forward_prop(X)[0]
    
    def one_run(self, X, Y):
        A2, cache = self.forward_prop(X)
        
        self.temps = cache
        
        grads = self.back_prop(A2, X, Y)
        self.grads = grads
        self.parameters = self.update_params()
        
    def forward_prop(self, X):
        parameters = self.parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = 1. / (1 + np.exp(-Z2))

        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

        return A2, cache
    
    def back_prop(self, A, X, Y):
        
        parameters = self.parameters
        cache = self.temps
        
        m = X.shape[1]

        W1 = parameters["W1"]
        W2 = parameters["W2"]

        b1 = parameters["b1"]
        b2 = parameters["b2"]

        A1 = cache["A1"]
        A2 = cache["A2"]

        dZ2 = A2 - Y
#         print(dZ2.shape, A1.T.shape)
        dW2 = (1. / m) * np.dot(dZ2, A1.T)
        db2 = (1. / m) * np.sum(dZ2, axis = 1, keepdims= True)
        # print(W2.T.shape, dZ2.shape, A1.shape)
        dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1. / m) * np.dot(dZ1, X.T)
        db1 = (1. / m) * np.sum(dZ1, axis = 1, keepdims=True)



        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

        return grads
        
        
    def update_params(self, learning_rate = 0.1):
        parameters = self.parameters
        grads = self.grads
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

        return parameters