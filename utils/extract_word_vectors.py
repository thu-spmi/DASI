'''
extrac word vectors that are pretrained on big corpus not on dictionary
'''
import json
import pickle, zipfile
import numpy as np
from gensim.models import Word2Vec, KeyedVectors

with open('./tmp_data/word_dict.json') as f:
    vocab_dict = json.load(f)

def extract_glove():
    # extract glove
    # download from https://nlp.stanford.edu/projects/glove/ and set the path by yourself
    # glove_path = 'D:\办公\Project_RLforDialogue\Experiment\glove.6B\glove.6B.300d.txt'
    glove_path = '/home/ouzj01/zhangyc/project/DASI/tmp_data/embeddings/glove.42B.300d.zip'

    glove_embs = {}
    # with open(glove_path, encoding='utf-8') as f:
    #     for l in f:
    #         line = l.split()
    #         glove_embs[line[0]] = np.fromstring('|'.join(line[1:]), dtype=np.float32, sep='|')

    archive = zipfile.ZipFile(glove_path, 'r')
    # glove_data = archive.open(glove_path.split('/')[-1][:-4]+'.txt', 'r').readlines()
    with archive.open(glove_path.split('/')[-1][:-4]+'.txt', 'r') as f:
    # print(glove_data[:100])
        for l in f:
            # print(l)
            line = l.decode("utf-8").split()
            # print(line)
            glove_embs[line[0]] = np.fromstring('|'.join(line[1:]), dtype=np.float32, sep='|')

    glove_embs_pretrain = {}
    for w in vocab_dict:
        if w in glove_embs:
            glove_embs_pretrain[w] = glove_embs[w]
        elif w.lower() in glove_embs:
            glove_embs_pretrain[w] = glove_embs[w.lower()]
        else:
            print('not contained in glove:', w)
    with open('../tmp_data/embeddings/glove42B_embs_big_corpus.pkl', 'wb') as f:
        pickle.dump(glove_embs_pretrain, f)

def extract_w2v():
    def load_bin_vec(fname, vocab):
        word_vecs = {}
        model = KeyedVectors.load_word2vec_format(fname, binary=True)
        for w in vocab:
            if w in model.wv:
                word_vecs[w] = model.wv[w]
        return word_vecs
    # download by your self
    vectors_file =  '../GoogleNews-vectors-negative300.bin'
    vectors = load_bin_vec(vectors_file, vocab_dict)  # pre-trained vectors
    with open('./tmp_data/embeddings/w2v_embs_big_corpus', 'wb') as f:
        pickle.dump(vectors, f)

def extract_paragram():
    # extract glove
    # download from https://nlp.stanford.edu/projects/glove/ and set the path by yourself
    # glove_path = 'D:\办公\Project_RLforDialogue\Experiment\glove.6B\glove.6B.300d.txt'
    paragram_path = '../paragram_300_sl999/paragram_300_sl999.txt'


    paragram_embs = {}
    with open(paragram_path, encoding='utf-8') as f:
        for l in f:
            line = l.split()
            if line[0] in vocab_dict:
                paragram_embs[line[0]] = np.fromstring('|'.join(line[1:]), dtype=np.float32, sep='|')
    with open('../tmp_data/embeddings/paragram_sl999.pkl', 'wb') as f:
        pickle.dump(paragram_embs, f)


def extract_chinese():
    with open('../raw_data/sgns.merge.word', encoding='utf-8') as f:
        print(f.readline())
        # 取前15000个词向量
        for line in f:
            w = line.split()[0]
            print(w)



if __name__ == '__main__':
    print('extracting glove')
    extract_glove()
    print('extracting word2vector')
    extract_w2v()
    print('extracting paragram')
    extract_paragram()
    # extract_chinese()