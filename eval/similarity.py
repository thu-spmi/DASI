# -*- coding: utf-8 -*-

"""
 Functions for fetching similarity datasets
"""

import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch

def fetch_MTurk():
    """
    Fetch MTurk dataset for testing attributional similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,

    References
    ----------
    Radinsky, Kira et al., "A Word at a Time: Computing Word Relatedness Using Temporal Semantic Analysis", 2011

    Notes
    -----
    Human labeled examples of word semantic relatedness. The data pairs were generated using an algorithm as
    described in the paper by [K. Radinsky, E. Agichtein, E. Gabrilovich, S. Markovitch.].
    Each pair of words was evaluated by 10 people on a scale of 1-5.

    Additionally scores were multiplied by factor of 2.
    """
    #data = _get_as_pd('https://www.dropbox.com/s/f1v4ve495mmd9pw/EN-TRUK.txt?dl=1',
    #                  'similarity', header=None, sep=" ").values
    data = pd.read_csv('raw_data/word_benchmark/MTURK-771.csv', header=None, sep=",").values
    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float))


def fetch_MEN(which="all", form="natural"):
    """
    Fetch MEN dataset for testing similarity and relatedness

    Parameters
    ----------
    which : "all", "test" or "dev"
    form : "lem" or "natural"

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores

    References
    ----------
    Published at http://clic.cimec.unitn.it/~elia.bruni/MEN.html.

    Notes
    -----
    Scores for MEN are calculated differently than in WS353 or SimLex999.
    Furthermore scores where rescaled to 0 - 10 scale to match standard scaling.

    The MEN Test Collection contains two sets of English word pairs (one for training and one for testing)
    together with human-assigned similarity judgments, obtained by crowdsourcing using Amazon Mechanical
    Turk via the CrowdFlower interface. The collection can be used to train and/or test computer algorithms
    implementing semantic similarity and relatedness measures.
    """
    if which == "dev":
        data = pd.read_csv('raw_data/word_benchmark/EN-MEN-LEM-DEV.txt', header=None, sep=" ")
    elif which == "test":
        data = pd.read_csv('raw_data/word_benchmark/EN-MEN-LEM-TEST.txt', header=None, sep=" ")
    elif which == "all":
        data = pd.read_csv('raw_data/word_benchmark/EN-MEN-LEM-TEST.txt', header=None, sep=" ")
    else:
        raise RuntimeError("Not recognized which parameter")

    if form == "natural":
        # Remove last two chars from first two columns
        data = data.apply(lambda x: [y if isinstance(y, float) else y[0:-2] for y in x])
    elif form != "lem":
        raise RuntimeError("Not recognized form argument")
    return Bunch(X=data.values[:, 0:2].astype("object"), y=data.values[:, 2:].astype(np.float) / 5.0)


def fetch_WS353(which="all"):
    """
    Fetch WS353 dataset for testing attributional and
    relatedness similarity

    Parameters
    ----------
    which : 'all': for both relatedness and attributional similarity,
            'relatedness': for relatedness similarity
            'similarity': for attributional similarity
            'set1': as divided by authors
            'set2': as divided by authors

    References
    ----------
    Finkelstein, Gabrilovich, "Placing Search in Context: The Concept Revisited†", 2002
    Agirre, Eneko et al., "A Study on Similarity and Relatedness Using Distributional and WordNet-based Approaches",
    2009

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of std of scores if available (for set1 and set2)
    """
    if which == "all":
        data = pd.read_csv('raw_data/word_benchmark/EN-WS353.txt', header=0, sep="\t")
    elif which == "relatedness":
        data = pd.read_csv('raw_data/word_benchmark/EN-WSR353.txt', header=None, sep="\t")
    elif which == "similarity":
        data = pd.read_csv('raw_data/word_benchmark/EN-WSS353.txt', header=None, sep="\t")
    else:
        raise RuntimeError("Not recognized which parameter")

    # We basically select all the columns available
    X = data.values[:, 0:2]
    y = data.values[:, 2].astype(np.float)

    # We have also scores
    if data.values.shape[1] > 3:
        sd = np.std(data.values[:, 2:15].astype(np.float), axis=1).flatten()
        return Bunch(X=X.astype("object"), y=y, sd=sd)
    else:
        return Bunch(X=X.astype("object"), y=y)


def fetch_RG65():
    """
    Fetch Rubenstein and Goodenough dataset for testing attributional and
    relatedness similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of std of scores if available (for set1 and set2)

    References
    ----------
    Rubenstein, Goodenough, "Contextual correlates of synonymy", 1965

    Notes
    -----
    Scores were scaled by factor 10/4
    """
    data = pd.read_csv('raw_data/word_benchmark/EN-RG-65.txt',header=None, sep="\t").values

    return Bunch(X=data[:, 0:2].astype("object"),
                 y=data[:, 2].astype(np.float) * 10.0 / 4.0)



def fetch_SimLex999(which='all'):
    """
    Fetch SimLex999 dataset for testing attributional similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,
        'sd': vector of sd of scores,
        'conc': matrix with columns conc(w1), conc(w2) and concQ the from dataset
        'POS': vector with POS tag
        'assoc': matrix with columns denoting free association: Assoc(USF) and SimAssoc333

    References
    ----------
    Hill, Felix et al., "Simlex-999: Evaluating semantic models with (genuine) similarity estimation", 2014

    Notes
    -----
     SimLex-999 is a gold standard resource for the evaluation of models that learn the meaning of words and concepts.
     SimLex-999 provides a way of measuring how well models capture similarity, rather than relatedness or
     association. The scores in SimLex-999 therefore differ from other well-known evaluation datasets
     such as WordSim-353 (Finkelstein et al. 2002). The following two example pairs illustrate the
     difference - note that clothes are not similar to closets (different materials, function etc.),
     even though they are very much related: coast - shore 9.00 9.10, clothes - closet 1.96 8.00
    """

    data = pd.read_csv('raw_data/word_benchmark/EN-SIM999.txt',sep="\t")

    # We basically select all the columns available
    X = data[['word1', 'word2']].values
    y = data['SimLex999'].values
    sd = data['SD(SimLex)'].values
    conc = data[['conc(w1)', 'conc(w2)', 'concQ']].values
    POS = data[['POS']].values
    assoc = data[['Assoc(USF)', 'SimAssoc333']].values

    if which == 'all':
        idx = np.asarray(range(len(X)))
    elif which == '333':
        idx = np.where(assoc[:,1] == 1)[0]
    else:
        raise ValueError("Subset of SL999 not recognized {}".format(which))

    return Bunch(X=X[idx].astype("object"), y=y[idx], sd=sd[idx], conc=conc[idx],
                 POS=POS[idx], assoc=assoc[idx])

def fetch_SimVerb3500(which='all'):
    """
    Fetch SimVerb3500 dataset for testing verb similarity

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with scores,

    References
    ----------
    Gerz, Daniela et al., "SimVerb-3500: A Large-Scale Evaluation Set of Verb Similarity", 2016

    Notes
    -----
    TODO
    """
    if which not in ['all', 'dev', 'test']:
        raise RuntimeError("Not recognized which parameter")

    file_map = {"dev": 'raw_data/word_benchmark/dev_SimVerb3500.txt',
               "test": 'raw_data/word_benchmark/test_SimVerb3500.txt'}

    data = pd.read_csv(file_map[which], header=None, sep=" ")
    return Bunch(X=data.values[:, 0:2].astype("object"), y=data.values[:, 2:].astype(np.float))

def fetch_SCWS():
    """
    Fetch SCWS dataset for testing similarity (with a context)

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object. Keys of interest:
        'X': matrix of 2 words per column,
        'y': vector with mean scores,
        'sd': standard deviation of scores

    References
    ----------
    Huang et al., "Improving Word Representations via Global Context and Multiple Word Prototypes", 2012

    Notes
    -----
    TODO
    """
    data = pd.read_csv('raw_data/word_benchmark/preproc_SCWS.txt', header=None, sep="\t")
    X = data.values[:, 0:2].astype("object")
    mean = data.values[:,2].astype(np.float)
    sd = np.std(data.values[:, 3:14].astype(np.float), axis=1).flatten()
    return Bunch(X=X, y=mean,sd=sd)
