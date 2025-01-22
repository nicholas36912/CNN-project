import tensorflow as tf
import numpy as np 

(trainImages, trainLabels), (testImages, testLabels) = tf.keras.datasets.cifar10.load_data()

X_train = trainImages [:1000]/255
Y_train = trainLabels [:1000]

num_classes = 10 

#One-hot encode to represent classes as a binary value in a vector(0-9)
def one_hot_encode(labels, num_classes):
    #initialize one-hot label matrix, each column a label, and rows of num classes
    one_hot_labels = np.zeros((len(labels), num_classes))
    #iterate through each label
    for i in range (len(labels)):
    #set corrospinding entry in the matrix to 1 
        one_hot_labels[i, labels[i]] = 1 
    return one_hot_labels

Y_train = one_hot_encode(Y_train, num_classes)



