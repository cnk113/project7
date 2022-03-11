# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
from collections import Counter
import random

# Defining preprocessing functions
def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one hot encoding of a list of nucleic acid sequences
    for use as input into a fully connected neural net.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence
            length due to the one hot encoding scheme for nucleic acids.

            For example, if we encode 
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    """
    encode = {'A':[1, 0, 0, 0], 'T':[0, 1, 0, 0], 'C':[0, 0, 1, 0], 'G':[0, 0, 0, 1]}
    row = []
    for s in seq_arr:
        encoding = np.array([encode[c] for c in s])
        row.append(encoding.flatten())
    return np.array(row)


def sample_seqs(
        seqs: List[str],
        labels: List[bool]) -> Tuple[List[str], List[bool]]:     
    """
    This function should sample your sequences to account for class imbalance. 
    Consider this as a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    counts = Counter(labels)
    oversampled = counts.most_common(1)[0][0]
    over_seq = []
    sampled_seqs = []
    sampled_labels = []
    for i in range(len(seqs)):
        if labels[i] == oversampled:
            over_seq.append(seqs[i])
        else:
            sampled_seqs.append(seqs[i]) # Undersampled seqs are added as is
            sampled_labels.append(labels[i])
    idx = random.sample(range(len(over_seq)), len(sampled_seqs))
    for i in idx:
        sampled_seqs.append(over_seq[i]) # Subsampling from majority
        sampled_labels.append(oversampled)
    return sampled_seqs, sampled_labels
