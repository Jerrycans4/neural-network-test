import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("mnist_test.csv")

data = np.array(data)
m, n = data.shape

np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z_new = np.copy(Z)
    Z_max = np.max(Z_new, axis=0, keepdims=True)
    Z_new = Z_new - Z_max
    Z_exp = np.exp(Z_new)
    csums = np.sum(Z_exp, axis=0)
    return Z_exp / csums

def forward_prop(W1, b1, W2, b2, X):
    #Hidden 
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)
    #Output
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    y_hot = np.zeros((Y.size, 10))
    y_hot[np.arange(Y.size), Y] = 1
    return y_hot.T

def ReLU_derivative(Z):
    return Z > 0

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    Y = one_hot(Y)
    dZ2 = A2 - Y
    dW2 = 1/Y.size * np.dot(dZ2, A1.T)
    db2 = 1/Y.size * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * ReLU_derivative(Z1)
    dW1 = 1/Y.size * np.dot(dZ1, X.T)
    db1 = 1/Y.size * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2 

def init_params():
    W1 = np.random.rand(10, 784) * np.sqrt(1./784)
    b1 = np.random.rand(10, 1) - 0.5 
    W2 = np.random.rand(10, 10) * np.sqrt(1./10)
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def update_params(W1, b1, W2, b2, alpha, dW1, db1, dW2, db2):
    W1 = W1 - (alpha * dW1)
    b1 = b1 - (alpha * db1)
    W2 = W2 - (alpha * dW2)
    b2 = b2 - (alpha * db2)
    return W1, b1, W2, b2

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, alpha, dW1, db1, dW2, db2)

        if i % 10 == 0:
            print(f"Iteration: {i}")
            predictions = get_predictions(A2)
            print("Accuracy: ", get_accuracy(predictions, Y))
    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def visualize_weights(W1):
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        weight_image = W1[i].reshape(28, 28)
        plt.imshow(weight_image, cmap='RdBu', interpolation='nearest')
        plt.title(f'Neuron {i}')
        plt.axis('off')
    plt.show()

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def visualize_failure(X, Y, W1, b1, W2, b2):
    predictions = make_predictions(X, W1, b1, W2, b2)
    failures = np.where(predictions != Y)[0]
    
    if len(failures) > 0:
        index = failures[0] 
        print(f"Index: {index}")
        print(f"Predicted: {predictions[index]}, Actual: {Y[index]}")
        
        current_image = X[:, index].reshape(28, 28)
        plt.gray()
        plt.imshow(current_image * 255, interpolation='nearest')
        plt.show()
    else:
        print("No failures found.")

if __name__ == '__main__':
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 2000, 0.1)
    print("#" * 20)
    visualize_weights(W1)
    visualize_failure(X_train, Y_train, W1, b1, W2, b2)
    print("#" * 20)
    dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
    print("Dev accuracy: ", get_accuracy(dev_predictions, Y_dev))

