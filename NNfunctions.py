from typing import List, Any
import math
import numpy as np

def sigmoid(v: float) -> float:
    """
    This is sigmoid function
    """
    return 1/(1+math.exp(-v))

def Nout(x: List[Any], w: List[Any]) -> float:
    """
    Sum of input,bias * weight(i)
    x=numpy.array() array of inputs and bias
    w=numpy.array() array of input weights and bias weight
    """
    return sum(np.multiply(x, w))

def gradOut(error:float, y:float) -> float:
    """gradient of output node
    diff activation fuction is sigmoid
    y*(1-y)
        e is error of the node
    y is the output of the node"""
    gradient = error * (y*(1-y))
    return gradient

def gradHidden(y:float, sum) -> float:
    """gradient of hidden node
    diff activation fuction is sigmoid
    y*(1-y)
    y is the output of the node
    sum is sum of previous nodes* weight"""
    gradient = (y*(1-y)) * sum
    return gradient


def deltaWeight(learningRate: float, gradient: float, x: float) -> float:
    """Calculate the delta weight
    l is learning rate
    g is gradient of the node
    x is input of the node"""
    d = -learningRate * gradient * x
    return (d)