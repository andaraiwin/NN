from typing import List, Any
import math
import numpy as np


def sigmoid(v: float) -> float:
    """
    This is sigmoid function
    """
    return 1 / (1 + math.exp(-v))


# Define the sigmoid activator; we ask if we want the sigmoid or its derivative
def sigmoid_act(x, der=False):
    if der == True:  # derivative of the sigmoid
        f = x * (1 - x)
    else:  # sigmoid
        f = 1 / (1 + np.exp(-x))

    return f


# We may employ the Rectifier Linear Unit (ReLU)
def ReLU_act(x, der=False):
    if der == True:
        if x.all() > 0:
            f = 1
        else:
            f = 0
    else:
        if x.all() > 0:
            f = x
        else:
            f = 0
    return f


# Now we are ready to define the perceptron;
# it eats a np.array (that may be a list of features )
def perceptron(X, act='Sigmoid'):
    shapes = X.shape  # Pick the number of (rows, columns)!
    n = shapes[0] + shapes[1]
    # Generating random weights and bias
    w = 2 * np.random.random(shapes) - 0.5  # We want w to be between -1 and 1
    b = np.random.random(1)

    # Initialize the function
    f = b[0]

    for i in range(0, X.shape[0] - 1):  # run over column elements
        for j in range(0, X.shape[1] - 1):  # run over rows elements
            f += w[i, j] * X[i, j] / n

    # Pass it to the activation function and return it as an output
    if act == 'Sigmoid':
        output = sigmoid_act(f)
    else:
        output = ReLU_act(f)

    return output


def Nout(x: List[Any], w: List[Any]) -> float:
    """
    Sum of input,bias * weight(i)
    x: numpy.array() array of inputs and bias
    w: numpy.array() array of input weights and bias weight
    """
    return sum(np.multiply(x, w))


def gradOut(error: float, y: float) -> float:
    """
    gradient of output node
    diff activation fuction is sigmoid
    thus, diff act func: y*(1-y)
    e is error of the node
    y is the output of the node
    """
    gradient: float = error * (y * (1 - y))
    return gradient


def gradHidden(y: float, sum: float) -> float:
    """
    gradient of hidden node
    diff activation fuction is sigmoid
    y*(1-y)
    y is the output of the node
    sum is sum of previous nodes* weight
    """
    gradient: float = (y * (1 - y)) * sum
    return gradient


def deltaWeight(learningRate: float, gradient: float, x: float) -> float:
    """
    Calculate the delta weight
    l is learning rate
    g is gradient of the node
    x is input of the node
    """
    d: float = -learningRate * gradient * x
    return d


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    data = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    data.drop(columns='bias', inplace=True)

    w1 = [np.random.uniform(-1, 1) for _ in range(0, 9)]
    # w1 = 2*np.random.rand(p, data.shape[1]) - 0.5
    w2 = [np.random.uniform(-1, 1) for _ in range(0, 9)]
    # w2 = 2*np.random.rand(p, data.shape[1]) - 0.5
    wOut = [np.random.uniform(-1, 1) for _ in range(0, 3)]
    # wOut = 2*np.random.rand(q)
    bias_out = 1

    print(f"W2: {w2}")

    # Feed forward propagation
    # input
    X = data.to_numpy()[5]

    # forward
    z1 = np.dot(w1, X)
    z2 = np.dot(w2, X)

    x_out = [z1, z2, bias_out]
    # output
    y = sigmoid_act(np.dot(wOut, x_out), der=False)

    # mapping y to class
    y = 2 if y < 0.5 else 4

    # Compute error
    error = (y_train.Class.to_numpy()[5] - y)
    # print(error)

    lr = 0.01
    # Backward propagation
    delta_Out = error * sigmoid_act(y, der=True)
    # print(deltaOut)
    for i in range(0, len(wOut)):
        delta_2 = sigmoid_act(z2, der=True) * delta_Out * wOut[i]  # First Layer backpropagation
        delta_1 = sigmoid_act(z1, der=True) * delta_Out * wOut[i]  # First Layer backpropagation

    # Update weight & bias output node