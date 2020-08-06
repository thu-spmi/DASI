import tensorflow as tf
import numpy as np
import os
import json, pickle
from input_data_ch import Dataset

class Run_CPAE_ch:
    def __init__(self):
        with open('../../tmp_data/xinhua_dict_afterseg.json', 'r', encoding='utf8') as f:
            self.rawdata = json.load(f)

        with open('../../tmp_data/chinese_dict.json', 'r', encoding='utf8') as f:
            self.vocab_dict = json.load(f)
            self.reversed_vocab_dict = dict(zip(self.vocab_dict.values(),
                                                self.vocab_dict.keys()))
        with open('../../tmp_data/chinese_syn.json', 'r', encoding='utf8') as f:
            self.syn = json.load(f)

        with open('../../tmp_data/chinese_ant.json', 'r', encoding='utf8') as f:
            self.ant = json.load(f)

        self.save_dir = '../../tmp_data/embeddings'
        self.output_filename = 'ch_CPAE_embs'
        self.max_length = 100
        self.vocab_size = len(self.vocab_dict)
        self.emb_size = 300
        self.hidden_size = 150
        self.max_grad_norm = 5
        self.lr = 0.001
        self.seed = 1234
        self.total_epochs = 5
        self.batch_size = 32
        self.CP_coef = 10
        self.syn_coef = 0.01
        self.retrofit_coef = 0.1
        self.is_pretrain = True
        with open('../../tmp_data/embeddings/ch_w2v_embs.pkl', 'rb') as f:
            self.pretrained_embs = pickle.load(f)

    def train(self):
        dataset = Dataset(self.rawdata, self.syn, self.ant)
        tf.set_random_seed(self.seed)  # for reproduce
        model = CPAE(
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
            reversed_vocab_dict=self.reversed_vocab_dict
        )
        with tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.group(tf.global_variables_initializer()))
            average_AE_loss = 0
            average_CP_loss = 0
            plot_every_steps = 500
            early_stopping = False
            for epoch in range(1, self.total_epochs + 1):
                if early_stopping: break
                steps = dataset.len // self.batch_size
                for step in range(1, steps+1):
                    if early_stopping: break
                    definition_, definition_seqlen_, defined_word_ = self.transform(
                        dataset.get_batch(self.batch_size))
                    synonym_word_from_, synonym_word_to_, synonym_flag_ = self.transform_syn(
                        dataset.get_batch_syn())
                    _, training_loss, CP_loss = sess.run([model.train_op,
                                                          model.AE_loss,
                                                          model.CP_loss],
                                                         feed_dict={
                                                             model.defined_word: defined_word_,
                                                             model.definition: definition_,
                                                             model.definition_seqlen: definition_seqlen_,
                                                             model.synonym_word_from: synonym_word_from_,
                                                             model.synonym_word_to: synonym_word_to_,
                                                             model.synonym_flag: synonym_flag_})
                    average_AE_loss += training_loss / plot_every_steps
                    average_CP_loss += CP_loss / plot_every_steps
                    if step % plot_every_steps == 0:
                        if self.CP_coef > 0:
                            print("epoch %2d, step %5d, AE_loss=%0.4f, CP_loss=%0.4f" %
                                  (epoch, step, average_AE_loss, average_CP_loss))
                        else:
                            print("epoch %2d, step %5d, AE_loss=%0.4f" %
                                  (epoch, step, average_AE_loss))

                        average_AE_loss = 0
                        average_CP_loss = 0
                average_AE_loss = 0
                average_CP_loss = 0
            print('save vectors ... ')
            cur_embs = self.another_test(sess, model)
            with open(os.path.join(self.save_dir, self.output_filename + '.pkl'), 'wb') as f:
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

    def transform_syn(self, batch_data):
        batch_size = len(batch_data)
        synonym_word_from = np.zeros(shape=[batch_size], dtype=np.int32)
        synonym_word_to = np.zeros(shape=[batch_size], dtype=np.int32)
        synonym_flag = np.zeros(shape=[batch_size], dtype=np.int32)
        for i, (w_from, w_to, w_flag) in enumerate(batch_data):
            synonym_word_from[i] = self.vocab_dict[w_from]
            synonym_word_to[i] = self.vocab_dict[w_to]
            synonym_flag[i] = w_flag
        return synonym_word_from, synonym_word_to, synonym_flag



class CPAE:
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
                 reversed_vocab_dict
                 ):
        self.name = name

        with tf.variable_scope(self.name,
             initializer=tf.truncated_normal_initializer(0, 0.03)):
            # model placeholder
            self.definition = tf.placeholder(dtype=tf.int32, shape=[None, max_length], name='def_sent')
            self.defined_word = tf.placeholder(dtype=tf.int32, shape=[None], name='defined_word')
            self.definition_seqlen = tf.placeholder(dtype=tf.int32, shape=[None], name='def_seqlen')
            self.batch_size = tf.shape(self.definition)[0]

            self.mask = tf.sequence_mask(self.definition_seqlen, max_length)
            # print(self.def_sent_mask)

            # 近义词 反义词  batchsize 和 defined words 可以不一样
            self.synonym_word_from = tf.placeholder(
                dtype=tf.int32, shape=[None], name='synonym_word_from')
            self.synonym_word_to = tf.placeholder(
                dtype=tf.int32, shape=[None], name='synonym_word_to')
            self.synonym_flag = tf.placeholder(
                dtype=tf.int32, shape=[None], name='synonym_flag')

            with tf.variable_scope("embedding"):
                if is_pretrain is False:
                    self.W_emb = tf.get_variable(
                        shape=[vocab_size, emb_size], name='W_emb', dtype=tf.float32)
                else:
                    print('using pretrain embeddings')
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

            with tf.variable_scope("synonym"):
                self.synonym_word_embs = tf.nn.embedding_lookup(self.W_emb, self.synonym_word_from)
                self.synonym_word_logits = tf.add(tf.matmul(self.synonym_word_embs, self.W_AE), self.b_AE)
                self.synonym_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.synonym_word_to,
                    logits=self.synonym_word_logits
                ) * tf.cast(self.synonym_flag, tf.float32))

            with tf.variable_scope('retrofitting'):
                self.def_input_emb_freeze = tf.nn.embedding_lookup(self.W_emb_freeze, self.definition)
                self.word_emb_freeze = tf.nn.embedding_lookup(self.W_emb_freeze, self.defined_word)
                self.synonym_word_embs_freeze = tf.nn.embedding_lookup(self.W_emb_freeze, self.synonym_word_from)
                self.retrofit_loss = tf.reduce_mean(tf.square(self.input_emb - self.def_input_emb_freeze)) + \
                                     tf.reduce_mean(tf.square(self.word_emb - self.word_emb_freeze)) + \
                                     tf.reduce_mean(tf.square(self.synonym_word_embs - self.synonym_word_embs_freeze))

            with tf.variable_scope("train"):
                if is_pretrain is False:
                    self.loss = self.AE_loss + CP_coef * self.CP_loss + \
                                syn_coef * self.synonym_loss + \
                                retrofit_coef * self.retrofit_loss
                else:
                    self.loss = self.AE_loss + CP_coef * self.CP_loss + \
                                syn_coef * self.synonym_loss
                self.tvars = tf.trainable_variables()
                self.grads = tf.gradients(self.loss, self.tvars)
                self.grads, _ = tf.clip_by_global_norm(
                    self.grads, max_grad_norm)

                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                # create training operation
                self.train_op = self.optimizer.apply_gradients(
                    zip(self.grads, self.tvars))

if __name__ == '__main__':
    run_model = Run_CPAE_ch()
    run_model.train()




