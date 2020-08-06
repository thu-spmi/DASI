from gensim.models import Word2Vec
import json,pickle,os

class Run_W2v:
    def __init__(self, **kwargs):
        with open(kwargs['vocab_path']) as f:
            self.vocab_dict = json.load(f)

        with open(kwargs['data_path']) as f:
            self.rawdata = json.load(f)


        self.save_dir = kwargs['embedding_path']
        self.word_emb_size = kwargs['emb_dim']
        self.output_filename = kwargs['output_filename']

    def train(self):
        corpus = []
        for w in self.rawdata:
            corpus.append(self.rawdata[w]['definitions'])
            for e in self.rawdata[w]['examples']:
                corpus.append(e[0].split())
        model = Word2Vec(corpus, size=self.word_emb_size,
                         window=5, min_count=0, workers=4, iter=30)

        emb_dict = {}
        for w in self.vocab_dict:
            if w in model.wv:
                emb_dict[w] = model.wv[w]

        with open(os.path.join(self.save_dir, self.output_filename+'.pkl'), 'wb') as f:
            pickle.dump(emb_dict, f)
