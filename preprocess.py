import pickle

# ef = open('new_pretrain_CPAE_embs_relatedness.pkl', 'rb')
# emb_mat = pickle.load(ef)
# emb_mat2 = {}
# for k,v in emb_mat.items():
#     emb_mat2[k.lower()] = v
# print(len(emb_mat2))
# pickle.dump(emb_mat2, open('new_pretrain_CPAE_embs_relatedness_lower.pkl','wb'))
# pickle.dump(emb_mat2, open('new_pretrain_CPAE_embs_relatedness_py2.pkl','wb'), protocol=2)

# ef = open('new_pretrain_CPAE_embs_similarity.pkl', 'rb')
# emb_mat = pickle.load(ef)
# emb_mat2 = {}
# for k,v in emb_mat.items():
#     emb_mat2[k.lower()] = v
# pickle.dump(emb_mat2, open('new_pretrain_CPAE_embs_similarity_lower.pkl','wb'))
# pickle.dump(emb_mat2, open('new_pretrain_CPAE_embs_similarity_py2.pkl','wb'), protocol=2)


# ef = open('counter_fitting_cpae.pkl', 'rb')
# emb_mat = pickle.load(ef)
# emb_mat2 = {}
# for k,v in emb_mat.items():
#     emb_mat2[k.lower()] = v
# pickle.dump(emb_mat2, open('counter_fitting_cpae_py2.pkl','wb'), protocol=2)

# ef = open('counter_fitting_glove.pkl', 'rb')
# emb_mat = pickle.load(ef)
# emb_mat2 = {}
# for k,v in emb_mat.items():
#     emb_mat2[k.lower()] = v
# pickle.dump(emb_mat2, open('counter_fitting_glove_py2.pkl','wb'), protocol=2)

# ef = open('CPAE_pretrain_embs_big_corpus.pkl', 'rb')
# emb_mat = pickle.load(ef)
# emb_mat2 = {}
# for k,v in emb_mat.items():
#     emb_mat2[k.lower()] = v
# pickle.dump(emb_mat2, open('CPAE_pretrain_embs_big_corpus_py2.pkl','wb'), protocol=2)

# ef = open('CPAE_pretrain_paragram_sl9999.pkl', 'rb')
# emb_mat = pickle.load(ef)
# emb_mat2 = {}
# for k,v in emb_mat.items():
#     emb_mat2[k.lower()] = v
# pickle.dump(emb_mat2, open('CPAE_pretrain_paragram_sl9999_py2.pkl','wb'), protocol=2)

# ef = open('CPAE_pretrain_paragram_sl999.pkl', 'rb')
# emb_mat = pickle.load(ef)
# emb_mat2 = {}
# for k,v in emb_mat.items():
#     emb_mat2[k.lower()] = v
# pickle.dump(emb_mat2, open('CPAE_pretrain_paragram_sl999_py2.pkl','wb'), protocol=2)


ef = open('tmp_data/embeddings/w2v_embs_big_corpus.pkl', 'rb')
emb_mat = pickle.load(ef)
emb_mat2 = {}
for k,v in emb_mat.items():
    emb_mat2[k] = v
pickle.dump(emb_mat2, open('tmp_data/embeddings/w2v_embs_big_corpus_py2.pkl','wb'), protocol=2)

ef = open('tmp_data/embeddings/glove_embs_big_corpus.pkl', 'rb')
emb_mat = pickle.load(ef)
emb_mat2 = {}
for k,v in emb_mat.items():
    emb_mat2[k] = v
pickle.dump(emb_mat2, open('tmp_data/embeddings/glove_embs_big_corpus_py2.pkl','wb'), protocol=2)

ef = open('tmp_data/embeddings/paragram_sl999.pkl', 'rb')
emb_mat = pickle.load(ef)
emb_mat2 = {}
for k,v in emb_mat.items():
    emb_mat2[k] = v
pickle.dump(emb_mat2, open('tmp_data/embeddings/paragram_sl999_py2.pkl','wb'), protocol=2)