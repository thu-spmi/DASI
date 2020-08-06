"""
this file is revised on the github project https://github.com/nmrksic/counter-fitting
"""
import os
import json, pickle
import math
from numpy.linalg import norm
from numpy import dot
import numpy
from copy import deepcopy
from eval.eval_model import evaluate_on_all

class Run_CF:
    def __init__(self, **kwargs):
        with open(kwargs['vocab_path']) as f:
            self.vocab_dict = json.load(f)
            self.reversed_vocab_dict = dict(zip(self.vocab_dict.values(),
                                                self.vocab_dict.keys()))
        #


        # For fast evaluation
        with open(os.path.join(
                os.path.split(kwargs['benchmark_path'])[0],
                'benchmark_words.json')) as f:
            self.benchmark_words = json.load(f)

        self.save_dir = kwargs['embedding_path']
        self.output_filename = kwargs['output_filename']
        self.vocab_size = len(self.vocab_dict)
        self.emb_size = kwargs['emb_dim']
        self.total_epochs = 10
        self.batch_size = kwargs['batch_size']

        if kwargs['pretrain_embs_path'] is None:
            raise ValueError('counter-fitting must set pretrain_embs_path')
        else:
            print('using pretrain embs', kwargs['pretrain_embs_path'])
            with open(os.path.join(
                    self.save_dir, kwargs['pretrain_embs_path']+'.pkl'), 'rb') as f:
                self.pretrained_embs = pickle.load(f)

    def train(self, *args):
        current_experiment = ExperimentRun(self.pretrained_embs)
        cur_embs = counter_fit(current_experiment)
        with open(os.path.join(self.save_dir, self.output_filename + '.pkl'), 'wb') as f:
            pickle.dump(cur_embs, f)
        results = evaluate_on_all(cur_embs)
        print(results)


class ExperimentRun:
    def __init__(self,
                 pretrained_word_vectors,
                 hyper_k1=0.1,
                 hyper_k2=0.1,
                 hyper_k3=0.1,
                 delta=1.0,
                 gamma=0.0,
                 rho=0.2,
                 ant_filepath = './tmp_data/antonyms.txt',
                 syn_filepath = './tmp_data/synonyms.txt'):

        # load pretrained word vectors and initialise their (restricted) vocabulary.
        self.pretrained_word_vectors = normalise_word_vectors(pretrained_word_vectors)
        self.vocabulary = set(self.pretrained_word_vectors.keys())
        self.synonyms = load_constraints(syn_filepath, self.vocabulary)
        self.antonyms = load_constraints(ant_filepath, self.vocabulary)
        self.hyper_k1 = hyper_k1
        self.hyper_k2 = hyper_k2
        self.hyper_k3 = hyper_k3
        self.delta = delta
        self.gamma = gamma
        self.rho = rho

def normalise_word_vectors(word_vectors, norm=1.0):
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word] ** 2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors


def load_constraints(constraints_filepath, vocabulary):
    constraints_filepath.strip()
    constraints = set()
    with open(constraints_filepath, "r+") as f:
        for line in f:
            word_pair = line.split()
            if word_pair[0] in vocabulary and word_pair[1] in vocabulary and word_pair[0] != word_pair[1]:
                constraints |= {(word_pair[0], word_pair[1])}
                constraints |= {(word_pair[1], word_pair[0])}
    return constraints


def distance(v1, v2, normalised_vectors=True):
    if normalised_vectors:
        return 1 - dot(v1, v2)
    else:
        return 1 - dot(v1, v2) / (norm(v1) * norm(v2))


def compute_vsp_pairs(word_vectors, vocabulary, rho=0.2):
    print("Pre-computing word pairs relevant for Vector Space Preservation (VSP). Rho =", rho)

    vsp_pairs = {}

    threshold = 1 - rho
    vocabulary = list(vocabulary)
    num_words = len(vocabulary)

    step_size = 1000  # Number of word vectors to consider at each iteration.
    vector_size = 300

    # ranges of word vector indices to consider:
    list_of_ranges = []

    left_range_limit = 0
    while left_range_limit < num_words:
        curr_range = (left_range_limit, min(num_words, left_range_limit + step_size))
        list_of_ranges.append(curr_range)
        left_range_limit += step_size

    range_count = len(list_of_ranges)

    # now compute similarities between words in each word range:
    for left_range in range(range_count):
        for right_range in range(left_range, range_count):

            # offsets of the current word ranges:
            left_translation = list_of_ranges[left_range][0]
            right_translation = list_of_ranges[right_range][0]

            # copy the word vectors of the current word ranges:
            vectors_left = numpy.zeros((step_size, vector_size), dtype="float32")
            vectors_right = numpy.zeros((step_size, vector_size), dtype="float32")

            # two iterations as the two ranges need not be same length (implicit zero-padding):
            full_left_range = range(list_of_ranges[left_range][0], list_of_ranges[left_range][1])
            full_right_range = range(list_of_ranges[right_range][0], list_of_ranges[right_range][1])

            for iter_idx in full_left_range:
                vectors_left[iter_idx - left_translation, :] = word_vectors[vocabulary[iter_idx]]

            for iter_idx in full_right_range:
                vectors_right[iter_idx - right_translation, :] = word_vectors[vocabulary[iter_idx]]

            # now compute the correlations between the two sets of word vectors:
            dot_product = vectors_left.dot(vectors_right.T)

            # find the indices of those word pairs whose dot product is above the threshold:
            indices = numpy.where(dot_product >= threshold)

            num_pairs = indices[0].shape[0]
            left_indices = indices[0]
            right_indices = indices[1]

            for iter_idx in range(0, num_pairs):

                left_word = vocabulary[left_translation + left_indices[iter_idx]]
                right_word = vocabulary[right_translation + right_indices[iter_idx]]

                if left_word != right_word:
                    # reconstruct the cosine distance and add word pair (both permutations):
                    score = 1 - dot_product[left_indices[iter_idx], right_indices[iter_idx]]
                    vsp_pairs[(left_word, right_word)] = score
                    vsp_pairs[(right_word, left_word)] = score
        return vsp_pairs


def vector_partial_gradient(u, v, normalised_vectors=True):
    if normalised_vectors:
        gradient = u * dot(u, v) - v
    else:
        norm_u = norm(u)
        norm_v = norm(v)
        nominator = u * dot(u, v) - v * numpy.power(norm_u, 2)
        denominator = norm_v * numpy.power(norm_u, 3)
        gradient = nominator / denominator

    return gradient


def one_step_SGD(word_vectors,
                 synonym_pairs,
                 antonym_pairs,
                 vsp_pairs,
                 current_experiment):

        new_word_vectors = deepcopy(word_vectors)

        gradient_updates = {}
        update_count = {}
        oa_updates = {}
        vsp_updates = {}

        # AR term:
        for (word_i, word_j) in antonym_pairs:

            current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

            if current_distance < current_experiment.delta:

                gradient = vector_partial_gradient(new_word_vectors[word_i], new_word_vectors[word_j])
                gradient = gradient * current_experiment.hyper_k1

                if word_i in gradient_updates:
                    gradient_updates[word_i] += gradient
                    update_count[word_i] += 1
                else:
                    gradient_updates[word_i] = gradient
                    update_count[word_i] = 1

        # SA term:
        for (word_i, word_j) in synonym_pairs:

            current_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

            if current_distance > current_experiment.gamma:

                gradient = vector_partial_gradient(new_word_vectors[word_j], new_word_vectors[word_i])
                gradient = gradient * current_experiment.hyper_k2

                if word_j in gradient_updates:
                    gradient_updates[word_j] -= gradient
                    update_count[word_j] += 1
                else:
                    gradient_updates[word_j] = -gradient
                    update_count[word_j] = 1

        # VSP term:
        for (word_i, word_j) in vsp_pairs:

            original_distance = vsp_pairs[(word_i, word_j)]
            new_distance = distance(new_word_vectors[word_i], new_word_vectors[word_j])

            if original_distance <= new_distance:

                gradient = vector_partial_gradient(new_word_vectors[word_i], new_word_vectors[word_j])
                gradient = gradient * current_experiment.hyper_k3

                if word_i in gradient_updates:
                    gradient_updates[word_i] -= gradient
                    update_count[word_i] += 1
                else:
                    gradient_updates[word_i] = -gradient
                    update_count[word_i] = 1

        for word in gradient_updates:
            # we've found that scaling the update term for each word helps with convergence speed.
            update_term = gradient_updates[word] / (update_count[word])
            new_word_vectors[word] += update_term

        return normalise_word_vectors(new_word_vectors)


def counter_fit(current_experiment):
    word_vectors = current_experiment.pretrained_word_vectors
    vocabulary = current_experiment.vocabulary
    antonyms = current_experiment.antonyms
    synonyms = current_experiment.synonyms

    current_iteration = 0

    vsp_pairs = {}

    if current_experiment.hyper_k3 > 0.0:  # if we need to compute the VSP terms.
        vsp_pairs = compute_vsp_pairs(word_vectors, vocabulary, rho=current_experiment.rho)

    # Post-processing: remove synonym pairs which are deemed to be both synonyms and antonyms:
    for antonym_pair in antonyms:
        if antonym_pair in synonyms:
            synonyms.remove(antonym_pair)
        if antonym_pair in vsp_pairs:
            del vsp_pairs[antonym_pair]

    max_iter = 10
    print("\nAntonym pairs:", len(antonyms), "Synonym pairs:", len(synonyms), "VSP pairs:", len(vsp_pairs))
    print("Running the optimisation procedure for", max_iter, "SGD steps...")

    while current_iteration < max_iter:
        current_iteration += 1
        word_vectors = one_step_SGD(word_vectors, synonyms, antonyms, vsp_pairs, current_experiment)
    return word_vectors
