import numpy as np
from formulas import activations
from layers import convolution, pooling, fully_connected
from data import datainit

def cross_entropy_loss(y_true, y_pred):
    n_samples = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    logp = -np.log(y_pred[range(n_samples), np.argmax(y_true, axis=1)])
    return np.sum(logp) / n_samples


# Initialize layers
conv1 = convolution.convolve3d(num_filters=3, filter_size=3, channels=3)
pool1 = pooling.maxpool(filter_size=2)
conv2 = convolution.convolve3d(num_filters=6, filter_size=3, channels=3)
pool2 = pooling.maxpool(filter_size=2)

# Dynamically calculate the flattened size for the dense layer
flattened_size = np.prod(pool2.forward(conv2.forward(pool1.forward(conv1.forward(datainit.X_train[0])))).shape)

dense1 = fully_connected.Layer_Dense(flattened_size, 64)
dense2 = fully_connected.Layer_Dense(64, 10)

def calculate_accuracy(X_test, Y_test):
    correct_predictions = 0
    total_predictions = X_test.shape[0]
    total_loss = 0

    for i in range(X_test.shape[0]):
        img = X_test[i]
        label = Y_test[i].reshape(1, -1)

        # Forward pass
        out = conv1.forward(img)
        out = pool1.forward(out)
        out = conv2.forward(out)
        out = pool2.forward(out)
        out_flat = out.flatten().reshape(1, -1)
        out_dense = dense1.forward(out_flat)
        out_softmax = activations.softmax(dense2.forward(out_dense))

        # Calculate loss
        loss = cross_entropy_loss(label, out_softmax)
        total_loss += loss

        # Calculate accuracy
        predicted_label = np.argmax(out_softmax, axis=1)
        true_label = np.argmax(label, axis=1)
        if predicted_label == true_label:
            correct_predictions += 1

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_predictions
    accuracy = correct_predictions / total_predictions * 100
    return avg_loss, accuracy