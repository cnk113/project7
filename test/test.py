# BMI 203 Project 7: Neural Network

# Import necessary dependencies here
from sklearn import datasets
from nn import preprocess
from collections import Counter
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
    pass


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

