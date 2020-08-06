from .similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_SimVerb3500, fetch_MTurk, fetch_RG65, fetch_SCWS
from .embedding import Embedding
import numpy as np
import scipy

def count_missing_words(w, X):
    missing_words = 0
    words = w.vocabulary.word_id
    for query in X:
        for query_word in query:
            if query_word not in words:
                missing_words += 1
    return missing_words

def cosine_similarity(v1, v2, model=None):
    """
    compute cosine similarity: regular between 2 vectors or max of pairwise or
    average of pairwise

    v1: ndarray of dimension 1 (dim,) or 2 (n_embeddings, dim)
    v2: ndarray of dimension 1 (dim,) or 2 (n_embeddings, dim)
    model: None if v1 and v2 have dimension 1
           else, 'AvgSim' or 'MaxSim'
    """
    if len(v1.shape) == 1:
        v1 = v1.reshape((1,-1))
    if len(v2.shape) == 1:
        v2 = v2.reshape((1,-1))
    prod_norm = np.outer(np.linalg.norm(v1, axis=1),np.linalg.norm(v2, axis=1))
    pairwise_cosine = np.dot(v1, v2.T)/prod_norm
    if not model:
        return pairwise_cosine[0][0]
    elif model == 'AvgSim':
        return np.mean(pairwise_cosine)
    elif model == 'MaxSim':
        return np.max(pairwise_cosine)
    else:
        return ValueError('Unknown model {}'.format(model))

def evaluate_similarity(w, X, y, missing_words = 'filter_out'):
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    n_missing_words = count_missing_words(w, X)
    # if n_missing_words > 0:
    #     print("Missing {} words.".format(n_missing_words))

    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    A, B = [], []
    if missing_words == 'mean' or n_missing_words == 0:
        A = [w.get(word, mean_vector) for word in X[:, 0]]
        B = [w.get(word, mean_vector) for word in X[:, 1]]
    elif missing_words == 'filter_out':
        # logger.info("Will ignore them")
        y_filtered = []
        for x, gt in zip(X, y):
            a, b = x
            if a not in w or b not in w:
                continue
            A.append(w.get(a, mean_vector))
            B.append(w.get(b, mean_vector))
            y_filtered.append(gt)
        y = np.asarray(y_filtered)

    #A = np.asarray([w.get(word, mean_vector) for word in X[:, 0]])
    #B = np.asarray([w.get(word, mean_vector) for word in X[:, 1]])
    scores = np.array([cosine_similarity(v1, v2) for v1, v2 in zip(A, B)])
    #scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    return scipy.stats.spearmanr(scores, y).correlation

def evaluate_on_all(w):
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    # Calculate results on similarity
    # print("Calculating similarity benchmarks")
    similarity_tasks = [
        ["SV-d", fetch_SimVerb3500(which='dev')],
        ["MEN-d", fetch_MEN(which='dev')],
        ["SL999", fetch_SimLex999()],
        ["SL333", fetch_SimLex999(which='333')],
        ["SV-t", fetch_SimVerb3500(which='test')],
        ["RG", fetch_RG65()],
        ["SCWS", fetch_SCWS()],
        ["MEN-t", fetch_MEN(which='test')],
        ["MT", fetch_MTurk()],
        ["WS353", fetch_WS353()],
    ]
    similarity_results = {}
    print_result = ''
    for name, data in similarity_tasks:
        similarity_results[name] = evaluate_similarity(w, data.X, data.y)
        print_result += "%s:%0.4f "%(name, similarity_results[name])
    return print_result


def get_validation(w):
    if isinstance(w, dict):
        w = Embedding.from_dict(w)
    data = fetch_SimVerb3500(which='dev')
    SV_score = evaluate_similarity(w, data.X, data.y)
    data =  fetch_MEN(which='dev')
    MEN_score = evaluate_similarity(w, data.X, data.y)
    data = fetch_SimLex999()
    # SL999 = evaluate_similarity(w, data.X, data.y)
    # same as CPAE paper, give more weight in similarity task
    return SV_score
    # return SL999
