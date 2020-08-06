import random, json

class Dataset:
    def __init__(self, wordnet, source, vocab_dict = None):
        self.data = [(k,v['definitions']) for k,v in wordnet.items()]
        self.len = len(self.data)
        self.batch_id = 0
        self.batch_id_syn = 0
        self.semantic_dict = {}
        self.synonyms_data = []

        if source == 'wordnet' or source == 'all':
            with open('tmp_data/syno_anto.json') as f:
                semantic_data = json.load(f)
            for k, v in semantic_data.items():
                for s in v[0]:
                    if k in vocab_dict and s in vocab_dict and k+'_'+s not in self.semantic_dict:
                        self.synonyms_data.append([k, s, 1])
                        self.semantic_dict[k+'_'+s] = 1
                for a in v[1]:
                    if k in vocab_dict and a in vocab_dict and k+'_'+a not in self.semantic_dict:
                        self.synonyms_data.append([k, a, -1])
                        self.semantic_dict[k+'_'+a] = 1

        if source == 'ppdb'  or source == 'all':
            with open('tmp_data/full-syn-sample.txt') as f:
                for syns in f.readlines():
                    w1, w2 = syns.split()
                    if w1 in vocab_dict and w2 in vocab_dict and w1+'_'+w2 not in self.semantic_dict:
                        self.synonyms_data.append([w1, w2, 1])
            with open('tmp_data/full-ant-sample.txt') as f:
                for atos in f.readlines():
                    w1, w2 = atos.split()
                    if w1 in vocab_dict and w2 in vocab_dict and w1+'_'+w2 not in self.semantic_dict:
                        self.synonyms_data.append([w1, w2, -1])


        random.shuffle(self.synonyms_data)
        print('synonyms_data size:', len(self.synonyms_data))

        # self.synonyms_data = []
        # self.synonyms = {}
        # self.atonyms = {}
        # with open(synonyms) as f:
        #     for syns in f.readlines():
        #         w1, w2 = syns.split()
        #         if w1 in vocab_dict and w2 in vocab_dict:
        #             if w1 not in self.synonyms:
        #                 self.synonyms[w1] = [w2]
        #             else:
        #                 self.synonyms[w1].append(w2)
        # with open(atonyms) as f:
        #     for atos in f.readlines():
        #         w1, w2 = atos.split()
        #         if w1 in vocab_dict and w2 in vocab_dict:
        #             if w1 not in self.atonyms:
        #                 self.atonyms[w1] = [w2]
        #             else:
        #                 self.atonyms[w1].append(w2)
        # # random.shuffle(self.synonyms_data)
        # # print('synonyms_data size:', len(self.synonyms_data))
        # self.batch_id_syn = 0



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


class Dataset_AttRep:
    def __init__(self, wordnet, source, vocab_dict = None):
        self.data = [(k,v['definitions']) for k,v in wordnet.items()]
        self.len = len(self.data) # 84418
        self.batch_id = 0
        self.batch_id_syn = 0
        self.batch_id_atn = 0
        self.semantic_dict = {}
        self.synonyms_data = []
        self.atonyms_data = []

        if source == 'wordnet' or source == 'all':
            with open('tmp_data/syno_anto.json') as f:
                semantic_data = json.load(f)
            for k, v in semantic_data.items():
                for s in v[0]:
                    if k in vocab_dict and s in vocab_dict and k+'_'+s not in self.semantic_dict:
                        self.synonyms_data.append([k, s])
                        self.semantic_dict[k+'_'+s] = 1
                for a in v[1]:
                    if k in vocab_dict and a in vocab_dict and k+'_'+a not in self.semantic_dict:
                        self.atonyms_data.append([k, a])
                        self.semantic_dict[k+'_'+a] = 1

        if source == 'ppdb'  or source == 'all':
            with open('linguistic_constraints/full-syn-sample.txt') as f:
                for syns in f.readlines():
                    w1, w2 = syns.split()
                    if w1 in vocab_dict and w2 in vocab_dict and w1+'_'+w2 not in self.semantic_dict:
                        self.synonyms_data.append([w1, w2])
            with open('linguistic_constraints/full-ant-sample.txt') as f:
                for atos in f.readlines():
                    w1, w2 = atos.split()
                    if w1 in vocab_dict and w2 in vocab_dict and w1+'_'+w2 not in self.semantic_dict:
                        self.atonyms_data.append([w1, w2])


        random.shuffle(self.synonyms_data)
        random.shuffle(self.atonyms_data)
        print('synonyms_data size:', len(self.synonyms_data))
        print('atonyms_data size:', len(self.atonyms_data))

        self.scale_factor = len(self.atonyms_data)/len(self.synonyms_data)


    def get_batch(self, batch_size):
        if self.batch_id + batch_size >= self.len:
            self.batch_id = 0
            random.shuffle(self.data)
        batch_data = self.data[self.batch_id:self.batch_id+batch_size]
        self.batch_id += batch_size
        return batch_data

    def get_batch_syn(self, batch_size_syn=6):
        batch_size_atn = int(batch_size_syn * self.scale_factor)
        if self.batch_id_syn + batch_size_syn >= len(self.synonyms_data):
            self.batch_id_syn = 0
            random.shuffle(self.synonyms_data)
        if self.batch_id_atn + batch_size_atn >= len(self.atonyms_data):
            self.batch_id_atn = 0
            random.shuffle(self.atonyms_data)

        batch_data_syn = self.synonyms_data[self.batch_id_syn:self.batch_id_syn + batch_size_syn]
        batch_data_atn = self.atonyms_data[self.batch_id_atn:self.batch_id_atn + batch_size_atn]
        self.batch_id_syn += batch_size_syn
        self.batch_id_atn += batch_size_atn
        return batch_data_syn, batch_data_atn