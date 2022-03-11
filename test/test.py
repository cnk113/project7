# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
from sklearn import datasets
from nn import preprocess, one_hot_encode_seqs 
from collections import Counter
import numpy as np
# TODO: Write your test functions and associated docstrings below.

def test_forward():
    pass


def test_single_forward():
    pass


def test_single_backprop():
    pass


def test_predict():
    pass


def test_binary_cross_entropy():
    pass


def test_binary_cross_entropy_backprop():
    pass


def test_mean_squared_error():
    pass


def test_mean_squared_error_backprop():
    pass


def test_one_hot_encode():
    '''
    Basic test for encoding
    '''
    s = preprocess.one_hot_encode_seqs(['A','T','C','G'])
    true = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    s2 = preprocess.one_hot_encode_seqs(['AGA'])
    true2 = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    assert np.allclose(s2, true2) and np.allclose(s,true), "Encoding failed"


def test_sample_seqs():
    '''
    Uses a binary class dataset to test if labels 50/50 for each class
    '''
    digits = datasets.load_breast_cancer()
    X = digits.data
    y = digits.target
    _, balanced_labels = preprocess.sample_seqs(X, y)
    counts = Counter(balanced_labels)
    assert counts.most_common()[0][1] == counts.most_common()[1][1], "Not balanced!"

