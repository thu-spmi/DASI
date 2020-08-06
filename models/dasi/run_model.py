import tensorflow as tf
import numpy as np
import os, logging, csv, random
import json, pickle, time
from .input_data import Dataset, Dataset_AttRep
from eval.eval_model import evaluate_on_all, get_validation

class Run_DASI:
    def __init__(self, **kwargs):
        with open(kwargs['vocab_path']) as f:
            self.vocab_dict = json.load(f)
            self.reversed_vocab_dict = dict(zip(self.vocab_dict.values(),
                                                self.vocab_dict.keys()))

        # For fast evaluation
        with open(os.path.join(
                os.path.split(kwargs['benchmark_path'])[0],
                'benchmark_words.json')) as f:
            self.benchmark_words = json.load(f)

        with open(kwargs['data_path']) as f:
            self.rawdata = json.load(f)
        self.model = kwargs['model']
        self.CP_coef = kwargs['CP_coef']
        self.retrofit_coef = kwargs['retrofit_coef']
        self.syn_coef = kwargs['syn_coef']
        self.emb_dir = kwargs['embedding_path']
        self.output_filename = kwargs['output_filename']
        self.max_length = kwargs['max_def_len']
        self.vocab_size = len(self.vocab_dict)
        self.emb_size = kwargs['emb_dim']
        self.hidden_size = kwargs['hidden_dim']
        self.max_grad_norm = kwargs['max_grad_norm']
        self.lr = kwargs['learning_rate']
        self.seed = kwargs['seed']
        self.total_epochs = kwargs['max_epochs']
        self.batch_size = kwargs['batch_size']
        self.pretrain_emb_path = kwargs['pretrain_embs_path']
        self.semantic = kwargs['semantic']
        self.use_emb = kwargs['use_emb']
        self.save_dir = kwargs['save_dir']
        self.sem_model = kwargs['sem_model']

        if kwargs['pretrain_embs_path'] is None:
            self.is_pretrain = False
            self.pretrained_embs = None
        else:
            self.is_pretrain = True
            # logging.info('using pretrain embs', kwargs['pretrain_embs_path'])
            with open(os.path.join(
                    self.emb_dir, kwargs['pretrain_embs_path']+'.pkl'), 'rb') as f:
                self.pretrained_embs = pickle.load(f)

        random.seed(self.seed)
        tf.set_random_seed(self.seed)  # for reproduce
        np.random.seed(self.seed)
        if self.model == 'CPAE':
            self.syn_coef = 0
            self.retrofit_coef = 0
        elif self.model == 'AE':
            self.CP_coef = 0
            self.syn_coef = 0
            self.retrofit_coef = 0



    def train(self, model='DASI'):
        if 'dasi' in self.sem_model:
            dataset = Dataset(self.rawdata, self.semantic, self.vocab_dict)
        else:
            dataset = Dataset_AttRep(self.rawdata, self.semantic, self.vocab_dict)
        self.dataset = dataset

        model = DASI(
            name=self.output_filename,
            max_length=self.max_length,
            vocab_size=self.vocab_size,
            emb_size=self.emb_size,
            hidden_size=self.hidden_size,
            max_grad_norm=self.max_grad_norm,
            learning_rate=self.lr,
            CP_coef=self.CP_coef,
            syn_coef=self.syn_coef,
            retrofit_coef=self.retrofit_coef,
            is_pretrain=self.is_pretrain,
            pretrained_embs=self.pretrained_embs,
            reversed_vocab_dict=self.reversed_vocab_dict,
            sem_model=self.sem_model
        )
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        # tf_config.gpu_options.allow_growth = True


        with tf.Session(config=tf_config) as sess:
            sess.run(tf.group(tf.global_variables_initializer()))
            average_AE_loss = 0
            average_CP_loss = 0
            average_attrep_loss = 0
            plot_every_steps = 30
            tolerance = 100
            tolerance_count = 0
            early_stopping = False
            warmup = 0
            max_valid_score = 0 # use validation benchmark to do early stopping
            start = time.time()
            for epoch in range(1, self.total_epochs + 1):
                if early_stopping: break
                steps = dataset.len // self.batch_size
                for step in range(1, steps+1):
                    if early_stopping: break
                    definition_, definition_seqlen_, defined_word_ = self.transform(
                        dataset.get_batch(self.batch_size))
                    if 'dasi' in self.sem_model:
                        word_from_, word_to_, flag_ = self.transform_syn(dataset.get_batch_syn(self.batch_size))
                        _, training_loss, CP_loss = sess.run([model.train_op,
                                                              model.AE_loss,
                                                              model.CP_loss],
                                                             feed_dict={
                                                                 model.defined_word: defined_word_,
                                                                 model.definition: definition_,
                                                                 model.definition_seqlen: definition_seqlen_,
                                                                 model.synonym_word_from: word_from_,
                                                                 model.synonym_word_to: word_to_,
                                                                 model.synonym_flag: flag_})
                    else:
                        batch_syn, batch_atn = dataset.get_batch_syn(self.batch_size)
                        syn_from, syn_to, atn_from, atn_to = self.transform_attrep(batch_syn, batch_atn)
                        _, training_loss, CP_loss, attrep_loss = sess.run([model.train_op,
                                                              model.AE_loss,
                                                              model.CP_loss,
                                                              model.attrep_loss],
                                                             feed_dict={
                                                                 model.defined_word: defined_word_,
                                                                 model.definition: definition_,
                                                                 model.definition_seqlen: definition_seqlen_,
                                                                 model.syn_from: syn_from,
                                                                 model.syn_to: syn_to,
                                                                 model.atn_from: atn_from,
                                                                 model.atn_to: atn_to})


                    average_AE_loss += training_loss / plot_every_steps
                    average_CP_loss += CP_loss / plot_every_steps
                    if step % plot_every_steps == 0:
                        now = (time.time()-start) / 60
                        if self.sem_model == 'attrep':
                            average_attrep_loss += attrep_loss / plot_every_steps
                            logging.info("epoch %2d, step %5d, AE_loss=%0.4f, CP_loss=%0.4f, attrep_loss=%0.4f, time: %.1f" %
                                  (epoch, step, average_AE_loss, average_CP_loss, average_attrep_loss, now))
                        elif self.CP_coef > 0:
                            logging.info("epoch %2d, step %5d, AE_loss=%0.4f, CP_loss=%0.4f, time: %.1f" %
                                  (epoch, step, average_AE_loss, average_CP_loss, now))
                        else:
                            logging.info("epoch %2d, step %5d, AE_loss=%0.4f, time: %.1f" %
                                  (epoch, step, average_AE_loss, now))
                        average_AE_loss = 0
                        average_CP_loss = 0
                        average_attrep_loss = 0

                        if epoch >= warmup:
                            if self.use_emb == 'definition':
                                cur_embs = self.test(sess, model)
                            else:
                                cur_embs = self.another_test(sess, model)
                            # cur_embs = self.test(sess, model)
                            cur_valid_score = get_validation(cur_embs)
                            # logging.info(evaluate_on_all(cur_embs))
                            if cur_valid_score>max_valid_score:
                                tolerance_count = 0
                                max_valid_score = cur_valid_score
                                results = evaluate_on_all(cur_embs)
                                logging.info(results)
                            else:
                                tolerance_count += 1
                                if tolerance_count == tolerance:
                                    early_stopping = True
                average_AE_loss = 0
                average_CP_loss = 0
            logging.info('best results under early stopping:')
            logging.info(results)
            results_dict = {'seed':self.seed, 'semantic': self.semantic, 'sem_model': self.sem_model,
                                    'emb': self.use_emb,
                                    'CP_coef': self.CP_coef, 'syn_coef': self.syn_coef,
                                    'retrofit_coef': self.retrofit_coef, 'epoch': epoch}
            for s in results.split():
                results_dict[s.split(':')[0]] = s.split(':')[1]

            res_path = './{}_{}_results.csv'.format(self.output_filename, self.pretrain_emb_path)
            write_title = False if os.path.exists(res_path) else True
            with open(res_path, 'a') as rf:
                writer = csv.DictWriter(rf, fieldnames=list(results_dict.keys()))
                if write_title:
                    writer.writeheader()
                writer.writerows([results_dict])
            with open(os.path.join(self.save_dir, 'embedding.pkl'), 'wb') as f:
                pickle.dump(cur_embs, f)

    def transform(self, batch_data):
        batch_size = len(batch_data)
        definition = np.zeros(shape=[batch_size, self.max_length], dtype=np.int32)
        definition_seqlen = np.zeros(shape=[batch_size], dtype=np.int32)
        defined_words = np.zeros(shape=[batch_size], dtype=np.int32)
        for i, (word, defs) in enumerate(batch_data):
            if word in self.vocab_dict:
                defined_words[i] = self.vocab_dict[word]
            for j, w in enumerate(defs):
                if w in self.vocab_dict and j < self.max_length:
                    definition[i, j] = self.vocab_dict[w]
            definition_seqlen[i] = len(defs) if len(defs) < self.max_length else self.max_length
        return definition, definition_seqlen, defined_words

    # def transform(self, batch_data):
    #     batch_size = len(batch_data)
    #     definition = np.zeros(shape=[batch_size, self.max_length], dtype=np.int32)
    #     definition_seqlen = np.zeros(shape=[batch_size], dtype=np.int32)
    #     defined_words = np.zeros(shape=[batch_size], dtype=np.int32)
    #     batch_dict = {}
    #     for i, (word, defs) in enumerate(batch_data):
    #         if word in self.vocab_dict:
    #             defined_words[i] = self.vocab_dict[word]
    #             batch_dict[word] = 1
    #         for j, w in enumerate(defs):
    #             if w in self.vocab_dict and j < self.max_length:
    #                 definition[i, j] = self.vocab_dict[w]
    #                 batch_dict[w] = 1
    #         definition_seqlen[i] = len(defs) if len(defs) < self.max_length else self.max_length

    #     word_num = len(batch_dict)
    #     # print('word number:', word_num)
    #     word_from = []
    #     word_to = []
    #     flag = []
    #     # synonym_word_from = np.zeros(shape=[batch_size], dtype=np.int32)
    #     # synonym_word_to = np.zeros(shape=[batch_size], dtype=np.int32)
    #     # synonym_flag = np.zeros(shape=[batch_size], dtype=np.int32)
    #     for w in batch_dict:
    #         synonyms = self.dataset.synonyms.get(w, [])
    #         atonyms = self.dataset.atonyms.get(w, [])
    #         for s in synonyms:
    #             word_from.append(self.vocab_dict[w])
    #             word_to.append(self.vocab_dict[s])
    #             flag.append(1)
    #         for a in atonyms:
    #             word_from.append(self.vocab_dict[w])
    #             word_to.append(self.vocab_dict[a])
    #             flag.append(-1)

    #     idx = list(range(batch_size))
    #     random.shuffle(idx)

    #     word_from = np.asarray(word_from, dtype=np.int32)[idx]
    #     word_to = np.asarray(word_to, dtype=np.int32)[idx]
    #     flag = np.asarray(flag, dtype=np.int32)[idx]
    #     # print("batch len:", len(flag))

    #     return definition, definition_seqlen, defined_words, word_from, word_to, flag

    def transform_syn(self, batch_data):
        batch_size = len(batch_data)
        word_from = np.zeros(shape=[batch_size], dtype=np.int32)
        word_to = np.zeros(shape=[batch_size], dtype=np.int32)
        flag = np.zeros(shape=[batch_size], dtype=np.int32)
        for i, (w_from, w_to, w_flag) in enumerate(batch_data):
            word_from[i] = self.vocab_dict[w_from]
            word_to[i] = self.vocab_dict[w_to]
            flag[i] = w_flag
        return word_from, word_to, flag


    def transform_attrep(self, batch_syn, batch_atn):
        batch_size_syn = len(batch_syn)
        batch_size_atn = len(batch_atn)
        syn_word_from = np.zeros(shape=[batch_size_syn], dtype=np.int32)
        syn_word_to = np.zeros(shape=[batch_size_syn], dtype=np.int32)
        atn_word_from = np.zeros(shape=[batch_size_atn], dtype=np.int32)
        atn_word_to = np.zeros(shape=[batch_size_atn], dtype=np.int32)
        for i, (w_from, w_to) in enumerate(batch_syn):
            syn_word_from[i] = self.vocab_dict[w_from]
            syn_word_to[i] = self.vocab_dict[w_to]
        for i, (w_from, w_to) in enumerate(batch_atn):
            atn_word_from[i] = self.vocab_dict[w_from]
            atn_word_to[i] = self.vocab_dict[w_to]
        return syn_word_from, syn_word_to, atn_word_from, atn_word_to


    def test(self, sess, model):
        '''
        优先输出 definition embs
        '''
        test_defined_words, test_undefined_words = [], []
        for w in self.vocab_dict:
            if w in self.rawdata:
                test_defined_words.append([w, self.rawdata[w]['definitions']])
            else:
                test_undefined_words.append(w)
        one_hop = 100
        times = len(test_defined_words) // one_hop
        definition_embs_all = {}
        for ii in range(times + 1):
            bdata = test_defined_words[ii * one_hop:(ii + 1) * one_hop]
            if not bdata: break
            definition_, definition_seqlen_, _ = self.transform(bdata)
            definition_emb = sess.run(model.definition_emb,
                                      feed_dict={
                                          model.definition: definition_,
                                          model.definition_seqlen: definition_seqlen_})
            for jj, emb in enumerate(definition_emb):
                definition_embs_all[bdata[jj][0]] = emb
        # 剩下的未定义的用input emb
        undefined_words_id = [self.vocab_dict[w] for w in test_undefined_words]
        undefined_words_emb = sess.run(model.word_emb,
                                       feed_dict={model.defined_word: undefined_words_id})  # 只是借用这个而已
        for ii, word_id in enumerate(undefined_words_id):
            definition_embs_all[self.reversed_vocab_dict[word_id]] = undefined_words_emb[ii]
        return definition_embs_all

    def another_test(self, sess, model):
        '''
        优先输出input embs
        '''
        definition_embs_all = {}
        undefined_words_id = [k for k in self.vocab_dict.values()]
        undefined_words_emb = sess.run(model.word_emb,
                                       feed_dict={model.defined_word: undefined_words_id})  # 只是借用这个而已
        for ii, word_id in enumerate(undefined_words_id):
            definition_embs_all[self.reversed_vocab_dict[word_id]] = undefined_words_emb[ii]
        return definition_embs_all




# DASI model contains AE, CPAE
class DASI:
    def __init__(self,
                 name,
                 max_length,
                 vocab_size,
                 emb_size,
                 hidden_size,
                 max_grad_norm,
                 learning_rate,
                 CP_coef,
                 syn_coef,
                 retrofit_coef,
                 is_pretrain,
                 pretrained_embs,
                 reversed_vocab_dict,
                 sem_model
                 ):
        self.name = name
        self.sem_model = sem_model

        with tf.variable_scope(self.name,
             initializer=tf.truncated_normal_initializer(0, 0.03)):
            # model placeholder
            self.definition = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='def_sent')
            self.defined_word = tf.placeholder(dtype=tf.int32, shape=[None], name='defined_word')
            self.definition_seqlen = tf.placeholder(dtype=tf.int32, shape=[None], name='def_seqlen')
            self.batch_size = tf.shape(self.definition)[0]

            self.mask = tf.sequence_mask(self.definition_seqlen, max_length)
            # logging.info(self.def_sent_mask)

            # 近义词 反义词  batchsize 和 defined words 可以不一样
            if 'dasi' in self.sem_model:
                self.synonym_word_from = tf.placeholder(
                    dtype=tf.int32, shape=[None], name='synonym_word_from')
                self.synonym_word_to = tf.placeholder(
                    dtype=tf.int32, shape=[None], name='synonym_word_to')
                self.synonym_flag = tf.placeholder(
                    dtype=tf.int32, shape=[None], name='synonym_flag')
            else:
                self.syn_from = tf.placeholder(
                    dtype=tf.int32, shape=[None], name='synonym_word_from')
                self.syn_to = tf.placeholder(
                    dtype=tf.int32, shape=[None], name='synonym_word_to')
                self.atn_from = tf.placeholder(
                    dtype=tf.int32, shape=[None], name='atonym_word_from')
                self.atn_to = tf.placeholder(
                    dtype=tf.int32, shape=[None], name='atonym_word_to')



            with tf.variable_scope("embedding"):
                if is_pretrain is False:
                    self.W_emb = tf.get_variable(
                        shape=[vocab_size, emb_size], name='W_emb', dtype=tf.float32)
                else:
                    logging.info('using pretrain embeddings')
                    W = np.random.randn(vocab_size, emb_size) * 0.03
                    for i in range(1, vocab_size):
                        if reversed_vocab_dict[i] in pretrained_embs:
                            W[i] = pretrained_embs[reversed_vocab_dict[i]]
                    self.W_emb = tf.Variable(initial_value=W, dtype=tf.float32, name='W_emb')
                    self.W_emb_freeze = tf.Variable(initial_value=W, dtype=tf.float32, name='W_emb_freeze',
                                                    trainable=False)
                self.input_emb = tf.nn.embedding_lookup(self.W_emb, self.definition)



            with tf.variable_scope("encoder"):
                self.encoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
                self.encoder_init_state = self.encoder_cell.zero_state(
                    batch_size=self.batch_size, dtype=tf.float32)

                # encode def_sent sentence
                _, self.output_state = tf.nn.dynamic_rnn(
                    cell=self.encoder_cell,
                    inputs=self.input_emb,
                    initial_state=self.encoder_init_state,
                    sequence_length=self.definition_seqlen,
                    dtype=tf.float32)
                self.definition_emb = tf.layers.dense(
                    self.output_state[1],
                    emb_size
                )

            self.W_AE = tf.get_variable(name='W_AE', shape=[emb_size, vocab_size], dtype=tf.float32)
            self.b_AE = tf.get_variable(name='b_AE', shape=[vocab_size], dtype=tf.float32)
            if self.sem_model == 'dasi_indp':
                self.W_AR = tf.get_variable(name='W_AR', shape=[emb_size, vocab_size], dtype=tf.float32)
                self.b_AR = tf.get_variable(name='b_AR', shape=[vocab_size], dtype=tf.float32)
            with tf.variable_scope("reconstruction"):
                self.predict_logits = tf.add(tf.matmul(self.definition_emb, self.W_AE), self.b_AE)
                self.predict_logits = tf.expand_dims(self.predict_logits, 1)
                self.batch_logits = tf.tile(self.predict_logits, [1, max_length, 1])
                self.batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.definition,
                    logits=self.batch_logits
                ) * tf.cast(self.mask, tf.float32)
                self.AE_loss = tf.reduce_sum(self.batch_loss) / tf.reduce_sum(
                    tf.cast(self.mask, tf.float32))

            with tf.variable_scope("consist_penalty"):
                self.word_emb = tf.nn.embedding_lookup(self.W_emb, self.defined_word)
                self.defined_word_mask = tf.cast(
                    tf.greater(self.defined_word,
                               tf.zeros_like(self.defined_word, dtype=tf.int32)), tf.float32)
                self.defined_wordemb_mask = tf.tile(
                    tf.expand_dims(self.defined_word_mask, 1), [1, emb_size])
                self.diff_embeddings = (self.word_emb - self.definition_emb) ** 2
                self.sum_proximity_term = tf.reduce_sum(
                    tf.reduce_mean(self.diff_embeddings * self.defined_wordemb_mask, axis=1))
                self.CP_loss = self.sum_proximity_term / (tf.reduce_sum(self.defined_word_mask) + 1e-9)

            if 'dasi' in self.sem_model:
                with tf.variable_scope("synonym"):
                    self.synonym_word_embs = tf.nn.embedding_lookup(self.W_emb, self.synonym_word_from)
                    if self.sem_model == 'dasi_indp':
                        self.synonym_word_logits = tf.add(tf.matmul(self.synonym_word_embs, self.W_AR), self.b_AR)
                    else:
                        self.synonym_word_logits = tf.add(tf.matmul(self.synonym_word_embs, self.W_AE), self.b_AE)
                    self.synonym_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.synonym_word_to,
                        logits=self.synonym_word_logits
                    ) * tf.cast(self.synonym_flag, tf.float32))

            elif self.sem_model == 'attrep':
                with tf.variable_scope("attact-repel"):
                    self.attract_margin, self.repel_margin = 1.0, 0.0

                    attract_examples_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_emb, self.syn_from), 1)
                    attract_examples_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_emb, self.syn_to), 1)
                    negative_examples_attract_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_emb, self.syn_from), 1)
                    negative_examples_attract_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_emb, self.syn_to), 1)
                    attract_similarity_between_examples = tf.reduce_sum(tf.multiply(attract_examples_left, attract_examples_right), 1)
                    attract_similarity_to_negatives_left = tf.reduce_sum(tf.multiply(attract_examples_left, negative_examples_attract_left), 1)
                    attract_similarity_to_negatives_right = tf.reduce_sum(tf.multiply(attract_examples_right, negative_examples_attract_right), 1)

                    self.attract_cost = tf.nn.relu(self.attract_margin + attract_similarity_to_negatives_left - attract_similarity_between_examples) + \
                                   tf.nn.relu(self.attract_margin + attract_similarity_to_negatives_right - attract_similarity_between_examples)

                    repel_examples_left = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_emb, self.atn_from), 1)
                    repel_examples_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_emb, self.atn_to), 1)
                    negative_examples_repel_left  = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_emb, self.atn_from), 1)
                    negative_examples_repel_right = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.W_emb, self.atn_to), 1)
                    repel_similarity_between_examples = tf.reduce_sum(tf.multiply(repel_examples_left, repel_examples_right), 1)
                    repel_similarity_to_negatives_left = tf.reduce_sum(tf.multiply(repel_examples_left, negative_examples_repel_left), 1)
                    repel_similarity_to_negatives_right = tf.reduce_sum(tf.multiply(repel_examples_right, negative_examples_repel_right), 1)

                    self.repel_cost = tf.nn.relu(self.repel_margin - repel_similarity_to_negatives_left + repel_similarity_between_examples) + \
                                   tf.nn.relu(self.repel_margin - repel_similarity_to_negatives_right + repel_similarity_between_examples)
                    self.attrep_loss = tf.reduce_mean(self.repel_cost) + tf.reduce_mean(self.attract_cost)

            if is_pretrain and self.sem_model != 'attrep':
                with tf.variable_scope('retrofitting'):
                    self.def_input_emb_freeze = tf.nn.embedding_lookup(self.W_emb_freeze, self.definition)
                    self.word_emb_freeze = tf.nn.embedding_lookup(self.W_emb_freeze, self.defined_word)
                    self.synonym_word_embs_freeze = tf.nn.embedding_lookup(self.W_emb_freeze, self.synonym_word_from)
                    self.retrofit_loss = tf.reduce_mean(tf.square(self.input_emb - self.def_input_emb_freeze)) + \
                                         tf.reduce_mean(tf.square(self.word_emb - self.word_emb_freeze)) + \
                                         tf.reduce_mean(tf.square(self.synonym_word_embs - self.synonym_word_embs_freeze))

            with tf.variable_scope("train"):
                if self.sem_model == 'attrep':
                    self.loss = self.AE_loss + \
                                    CP_coef * self.CP_loss + \
                                    syn_coef * self.attrep_loss
                else:
                    if is_pretrain is True:
                        self.loss = self.AE_loss + \
                                    CP_coef * self.CP_loss + \
                                    syn_coef * self.synonym_loss + \
                                    retrofit_coef * self.retrofit_loss
                    else:
                        self.loss = self.AE_loss + \
                                    CP_coef * self.CP_loss + \
                                    syn_coef * self.synonym_loss
                self.tvars = tf.trainable_variables()
                self.grads = tf.gradients(self.loss, self.tvars)
                self.grads, _ = tf.clip_by_global_norm(
                    self.grads, max_grad_norm)

                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                # create training operation
                self.train_op = self.optimizer.apply_gradients(
                    zip(self.grads, self.tvars))





