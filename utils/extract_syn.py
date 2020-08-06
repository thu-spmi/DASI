import json
with open('./tmp_data/word_dict.json') as f:
    vocab_dict = json.load(f)

#返回一个单词的同义词和反义词列表
from nltk.corpus import wordnet as wn
def Word_synonyms_and_antonyms(word):
    synonyms=[]
    antonyms=[]
    list_good=wn.synsets(word)
    for syn in list_good:
        for l in syn.lemmas():
            if l.name() in vocab_dict and l.name() != word:
                synonyms.append(l.name())
            if l.antonyms():
                if l.antonyms()[0].name() in vocab_dict:
                    antonyms.append(l.antonyms()[0].name())
    return [list(set(synonyms)),list(set(antonyms))]

syn_anto = dict()
size = 0
sem_count = 0
for w in vocab_dict:
    syn_anto[w] = Word_synonyms_and_antonyms(w)
    size += len(syn_anto[w][0]) + len(syn_anto[w][1])
    if sem_count
print(size)
with open('./tmp_data/syno_anto.json', 'w') as f:
    json.dump(syn_anto, f, indent=2)