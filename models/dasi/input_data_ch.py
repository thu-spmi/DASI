import random
random.seed(123)
from itertools import combinations

class Dataset:
    def __init__(self, xinhua, syn, ant):
        self.data = [(k, v) for k, v in xinhua.items()]
        self.len = len(self.data)
        print('data len:', self.len)
        self.batch_id = 0
        self.synonyms_data = []
        for item in syn:
            for w1, w2 in combinations(item, 2):
                self.synonyms_data.append([w1, w2, 1])
        for w1, w2 in ant:
            self.synonyms_data.append([w1, w2, -1])
        random.shuffle(self.synonyms_data)
        print('synonyms_data size:', len(self.synonyms_data))
        self.batch_id_syn = 0

    def get_batch(self, batch_size):
        if self.batch_id + batch_size >= self.len:
            self.batch_id = 0
            random.shuffle(self.data)
        batch_data = self.data[self.batch_id:self.batch_id+batch_size]
        self.batch_id += batch_size
        return batch_data

    def get_batch_syn(self, batch_size=6):
        if self.batch_id_syn + batch_size >= len(self.synonyms_data):
            self.batch_id_syn = 0
            random.shuffle(self.synonyms_data)
        batch_data = self.synonyms_data[self.batch_id_syn:self.batch_id_syn + batch_size]
        self.batch_id_syn += batch_size
        return batch_data
