import numpy as np

def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    logp = -np.log(y_pred[range(n_samples), np.argmax(y_true, axis=1)])
    return np.sum(logp) / n_samples