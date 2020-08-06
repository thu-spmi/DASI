from .config_registry import ConfigRegistry

configs = ConfigRegistry()

configs.set_root_config({
    # path where save training data
    'data_path': '',
    # path for word dictionary, we fix the dict rather than fix the number of words
    # because we want to contain benchmark_words, num_of_words = len(vocab_dict)
    'vocab_path': 'tmp_data/word_dict.json',
    # we need word benchmark valid data to early stop and evaluation
    'benchmark_path': 'raw_data/word_benchmark',
    # path for save generated embeddings
    'embedding_path': 'tmp_data/embeddings',
    # 'embedding_path': 'log/attrep',
    'synonym_path': 'tmp_data/syno_anto.json',
    # max definition length
    'max_def_len' : 100,
    # max example length
    'max_exam_len': 50,
    'batch_size' : 256,
    'emb_dim' : 300,
    'hidden_dim': 150,
    'decoder': 'skip-gram',
    # Optimizer is adam.
    'learning_rate' : 0.001,
    'max_grad_norm' : 5.0,
    'CP_coef': 10,  # the coefficient for consistency penalty term
    'syn_coef': 0.05,  # the coefficient for semantic injection penalty term
    'retrofit_coef': 0.05, # the coefficient for l2 loss between pre-trained and post-trained embs
    'max_epochs' : 100,
    'seed': 1,
    'pretrain_embs_path': None
})

c = configs['root']
# =============== dict only ==============
# glove model
c['data_path'] = 'tmp_data/corpus_wordnet'
c['output_filename'] = 'glove_embs_dict_only'
configs['glove_dict_only'] = c

# w2v model
c['data_path'] = 'tmp_data/wordnet'
c['output_filename'] = 'w2v_embs_dict_only'
configs['w2v_dict_only'] = c

# AE model
c['output_filename'] = 'AE_embs_dict_only'
configs['AE_dict_only'] = c

# CPAE model
c['output_filename'] = 'CPAE_embs_dict_only'
configs['CPAE_dict_only'] = c

# DASI model
c['output_filename'] = 'DASI_embs_dict_only'
configs['DASI_dict_only'] = c

# AE-pretrain model
# in this setting for all pretrain embs I use word2vec same in the CPAE paper
# c['pretrain_embs_path'] = 'w2v_embs_dict_only'  # you can change into glove if you want
c['output_filename'] = 'AE_pretrain_embs_dict_only'
configs['AE_pretrain_dict_only'] = c

# counter-fitting model
c['output_filename'] = 'CF_pretrain_embs_dict_only'
configs['CF_pretrain_dict_only'] = c

# CPAE-pretrain model
c['output_filename'] = 'CPAE_pretrain_embs_dict_only'
configs['CPAE_pretrain_dict_only'] = c

# DASI-pretrain model
c['output_filename'] = 'DASI_pretrain_embs_dict_only'
configs['DASI_pretrain_dict_only'] = c

# =============== big corpus ==============
# AE-pretrain model
# remenber set emb_size=300
# c['pretrain_embs_path'] = 'w2v_embs_big_corpus' # you can change into glove or paramgram-sl999 as you want
# c['pretrain_embs_path'] = 'glove_embs_big_corpus'

c['output_filename'] = 'AE_pretrain_embs_big_corpus'
configs['AE_pretrain_big_corpus'] = c

# CPAE-pretrain model
c['output_filename'] = 'CPAE_pretrain_embs_big_corpus'
configs['CPAE_pretrain_big_corpus'] = c

# counter-fitting model
c['output_filename'] = 'CF_pretrain_embs_big_corpus'
configs['CF_pretrain_big_corpus'] = c

# DASI-pretrain model
c['output_filename'] = 'DASI_pretrain_embs_big_corpus'
configs['DASI_pretrain_big_corpus'] = c


if __name__ == '__main__':
    configs.show()
