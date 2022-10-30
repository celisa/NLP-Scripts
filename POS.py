"""
POS Tagging Function to generate the components for HMMs

Elisa Chen, 2022
"""

import numpy as np
from viterbi import viterbi
import nltk


def get_pi(tags, states_set):
    """A function to generate the initial state distribution."""
    # get initial state distribution
    pi = []

    tags = np.array(tags)
    (unique, counts) = np.unique(tags, return_counts=True)
    unique = list(unique)
    counts = list(counts)
    for state in states_set:
        pi.append((counts[unique.index(state)] + 1) / (len(tags) + len(states_set)))
    return pi


def get_transition_matrix(tags, states_set):
    """A function to generate the transition matrix."""
    A = np.zeros((len(set(tags)), len(set(tags))))

    prev_state = tags[0]

    for i in range(1, len(tags)):
        curr_state = tags[i]
        A[
            states_set[prev_state], states_set[curr_state]
        ] += 1  # rows are previous states, columns are current states
        prev_state = curr_state

    # perform smoothing
    A = A + 1

    # normalize the values in the matrix
    A = A / A.sum(axis=1)[:, np.newaxis]
    return A


def get_observation_matrix(tags, words, states_set, words_set):
    """A function to generate the observation matrix."""

    # initialize observation matrix
    B = np.zeros((len(states_set), len(words_set)))

    for i in range(len(words)):
        B[states_set[tags[i]], words_set[words[i]]] += 1

    # perform smoothing?
    B = B + 1

    # normalize the values in the matrix
    B = B / B.sum(axis=1)[:, np.newaxis]
    return B


def create_dict(set_w):
    """A function to create a dictionary that maps strings onto ints."""
    dict = {}

    for ind in range(len(set_w)):
        dict[set_w[ind]] = ind
    return dict


def generate_POS_components():
    """A function to generate the transition matrix, observation matrix, and initial state distribution."""

    # get the training data
    nltk.download("brown")
    nltk.download("universal_tagset")
    train_data = nltk.corpus.brown.tagged_sents(tagset="universal")[:10000]
    train_data = [word for sentence in train_data for word in sentence]
    words = list(zip(*train_data))[0]
    tags = list(zip(*train_data))[1]

    states_set = list(set(tags))
    states_set.sort()
    states_set = create_dict(states_set)

    words_set = list(set(words))
    words_set.sort()
    words_set = create_dict(words_set)
    # handle unknown words
    words_set["UNK"] = len(words_set)

    # get initial state distribution
    pi = get_pi(tags, states_set)

    # get transition matrix
    A = get_transition_matrix(tags, states_set)

    # get observation matrix
    B = get_observation_matrix(tags, words, states_set, words_set)

    return pi, A, B, states_set, words_set


def get_obs_index(obs, words_ord):
    """A function to get the index of the observation in the observation matrix."""
    obs_indicies = []
    for word in obs:
        if word not in words_ord:
            # unknown word
            obs_indicies.append(words_ord["UNK"])
        else:
            obs_indicies.append(words_ord[word])
    return obs_indicies


def get_key(val, dict):
    """A function to get the key of a dictionary given the value."""
    for key, value in dict.items():
        if val == value:
            return key


def return_diff(pred, act):
    """A function to return the difference between the predicted and actual tags."""
    return [(pred[i], act[i], i) for i in range(len(pred)) if pred[i] != act[i]]


def run_tests(obs):
    """A function to provide POS tags for test input"""

    # flatten obs list
    obs = [word for sentence in obs for word in sentence]
    test_words = list(zip(*obs))[0]
    act_tags = list(zip(*obs))[1]
    pi, A, B, states_ord, words_ord = generate_POS_components()
    obs = get_obs_index(test_words, words_ord)
    pos_tags = viterbi(obs, pi, A, B)[0]
    pred_tags = [get_key(ind, states_ord) for ind in pos_tags]
    diff = return_diff(pred_tags, act_tags)
    return pred_tags, diff


if __name__ == "__main__":
    obs = nltk.corpus.brown.tagged_sents(tagset="universal")[10150:10153]
    pred_tags, diff = run_tests(obs)
    print(pred_tags)
