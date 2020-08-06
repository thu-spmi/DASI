# Visualize the embeddings.
from pylab import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# compare two embeddings
def plot_with_labels(embs, labels, filename):
    plt.figure()
    for i, label in enumerate(labels):
        x, y = embs[i, :]
        if label in ['feel', 'touch', 'bar', 'pub', 'beer']:
            plt.scatter(x, y, rcParams['lines.markersize'],  c='r')
        else:
            plt.scatter(x, y, rcParams['lines.markersize'], c='b')
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom',)
    plt.xticks([])
    plt.yticks([])
    plt.title(filename)
    plt.show()

def plotting(plot_words, embs, vocab_dict, reversed_vocab_dict, filename):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_embs = [embs[w] for w in plot_words]
    low_dim_embs = tsne.fit_transform(plot_embs)
    labels = [reversed_vocab_dict[vocab_dict[w]] for w in plot_words]
    plot_with_labels(low_dim_embs, labels, filename)

if __name__ == '__main__':
    import json
    import pickle
    with open('./tmp_data/word_dict.json') as f:
        vocab_dict = json.load(f)
    with open('./tmp_data/embeddings/CPAE_pretrain_embs_dict_only.pkl', 'rb') as f:
        embs = pickle.load(f)
    filename = './tmp_data/CPAE_dict_only.png'
    reversed_vocab_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    test_words = [
        'mother','son', 'daughter','father', 'child', 'children', 'wife', 'husband',
        'expensive', 'cheap', 'inexpensive', 'moderate',
        'east', 'north', 'west', 'south',
        'eastern', 'northern', 'western', 'southern',
        'bar', 'pub', 'coffee', 'beer', 'tea',
        'knit', 'crochet',
        'apple', 'orange', 'banana', 'grape', 'fruit',
        'pregnancy', 'pregnant',
        'fog', 'mist',
        # 'smart', 'intelligent',
        'hard', 'easy', 'difficult', 'simple',
        'fast', 'rapid',
        'immoral', 'bad',
        'unnecessary', 'necessary',
        'feel', 'touch',
        # 'insubordinate', 'defiant',
        'comparing', 'compare'
    ]
    plotting(test_words, embs, vocab_dict, reversed_vocab_dict, 'CPAE embeddings')
