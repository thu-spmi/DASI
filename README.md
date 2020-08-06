# DASI
Code and data for INTERSPEECH 2020 paper "Improved learning of Word Embeddings with Dictionary Definitions and Semantic Injection"

## Evironment
```
python3
tensorflow >= 1.12
nltk >= 3.2.5
pandas >= 0.22.0
scikit-learn >= 0.19.2
matplotlib
gensim
```

## Steps
1. Unzip files in tmp_data and raw_data
2. Preprocess wordnet data & build vocabulary    
    using `python utils/extract_data.py`  
   get synonyms and antonyms from wordnet using  
    `python  utils/extract_syn.py`
3. Run different models under different settings. An example: 
```
python run.py --model=DASI --config=DASI_pretrain_big_corpus --pretrain=w2v_embs_big_corpus --seed=1 --syn_coef=1.0 --retrofit_coef=0 --CP_coef=25 --batch_size=256 --semantic=all --use_emb=definition --sem_model=dasi
```
