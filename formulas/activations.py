import numpy as np

class Activations:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_backward(dvalues, inputs):
        dvalues[inputs <= 0] = 0
        return dvalues

    @staticmethod
    def softmax(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def softmax_backward(dvalues, y_true):
        n_samples = y_true.shape[0]
        dvalues[range(n_samples), np.argmax(y_true, axis=1)] -= 1
        dvalues /= n_samples
        return dvalues