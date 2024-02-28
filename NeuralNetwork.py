from LinearAlgebra import *
import sys
from typing import List
import math

# Import other modules using package sys
sys.path.insert(0, 'C:\\Anda\\data\\DataMining\\NeuralN\\LinearAlgebra.py')


def sigmoid(z: float) -> float:
    """Return sigmoid function"""
    return 1 / (1 + math.exp(-z))


def neuron_output(w: Vector, x: Vector) -> float:
    # weights includes the bias term, inputs includes a 1
    return sigmoid(dot(x, w))


def argmax(xs: list) -> int:
    """Returns the index of the largest value"""
    return max(range(len(xs)), key=lambda i: xs[i])


def neuron_hidden(w: Vector, x: Vector) -> None:
    result = w + x


if __name__ == "__main__":
    from LinearAlgebra import *

    w1: Vector = [1, 2, 3]

    print(argmax(w1))
