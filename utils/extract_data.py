'''
preprocessing wordnet data and generate dictionary
'''

import json
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import wordnet as wn
stop_words = stopwords.words('english')

# I use this size for all experiments
vocab_size = 20000

with open('raw_data/wordnet.json') as f:
    wordnet = json.load(f)

with open('raw_data/benchmark_words.json') as f:
    benchmark_words = set(json.load(f))
print('benchmark_words', len(benchmark_words))

with open('tmp_data/vocab_woz.json') as f:
    woz_words = set(json.load(f)['index2word'])
print('woz_words', len(woz_words))

total_words = benchmark_words | woz_words
print('Total_words', len(total_words))

corpus_wordnet = []
full_wordnet = {}
all_words = list(woz_words)*10

print('extract raw wordnet data ... ')
max_exam_len = 0
max_def_len = 0
for word, Lemmas in wordnet.items():
    if '_' not in word and word not in stop_words: # 去停词和词组
        defs = []
        examples = []
        for Lemma, items in Lemmas.items():
            definition = items['definition'].split()
            # <SEP> has two functions: seperate defs and
            # use as sentinel for examples, if target word not in examples,
            # predict the position of <SEP>
            defs += definition + ['<SEP>']
            corpus_wordnet += defs
            for example in items['examples']:
                all_words += example.split()
                corpus_wordnet += example.split()
                corpus_wordnet += ['<SEP>']
                max_exam_len = max(max_exam_len, len(example.split()))
            examples.extend([(ii[0]+' <SEP>', ii[1][0]) for ii in zip(items['examples'], items['targets'])])
        defs.pop()
        max_def_len = max(max_def_len, len(defs))
        full_wordnet[word] = {}
        full_wordnet[word]['definitions'] = defs
        full_wordnet[word]['examples'] = examples
        all_words += defs

with open('tmp_data/wordnet', 'w') as dst:
    json.dump(full_wordnet, dst, indent=2)
print('total defined words:', len(full_wordnet))
print('max_def_len', max_def_len)
print('max_exam_len', max_exam_len)

print('total in-definition words:', len(set(all_words)))
all_words += list(full_wordnet.keys())
print('after add defined words:', len(set(all_words)))

# generate dictionary
print('build word dictionary ... ')
word_dict = {}
word_dict['<UNK>'] = 0
common_words = Counter(all_words)
# print(common_words)
dic = dict(common_words)
print('receiving' in dic)
print('strangers' in dic)

common_benchmark_words = set(w for w in benchmark_words if w in dic)
common_total_words = set(w for w in total_words if w in dic)
print('number of benchmark words in dictionary: ', len(common_benchmark_words))
print('number of total words in dictionary', len(common_total_words))


# print(common_benchmark_words)
for item in common_words:
    word_dict[item] = len(word_dict)
    if item in common_total_words:
        common_total_words.remove(item)
    if len(word_dict) == vocab_size-len(common_total_words):
        for w in common_total_words:
            word_dict[w] = len(word_dict)
        break
with open('tmp_data/word_dict.json','w') as f:
    json.dump(word_dict, f)


with open('tmp_data/corpus_wordnet','w') as f:
    for i in range(len(corpus_wordnet)):
        if corpus_wordnet[i] not in word_dict:
            corpus_wordnet[i] = '<UNK>'
    f.write(' '.join(corpus_wordnet))


#返回一个单词的同义词和反义词列表
def Word_synonyms_and_antonyms(word):
    synonyms=[]
    antonyms=[]
    list_good=wn.synsets(word)
    for syn in list_good:
        for l in syn.lemmas():
            if l.name() in word_dict and l.name() != word:
                synonyms.append(l.name())
            if l.antonyms():
                if l.antonyms()[0].name() in word_dict:
                    antonyms.append(l.antonyms()[0].name())
    return [list(set(synonyms)),list(set(antonyms))]

syn_anto = dict()
size = 0
word_both = 0
word_onlydef = 0
word_onlysem = 0
for w in word_dict:
    syn_anto[w] = Word_synonyms_and_antonyms(w)
    size += len(syn_anto[w][0]) + len(syn_anto[w][1])
    has_sr = syn_anto[w][0] or syn_anto[w][1]
    if w in common_benchmark_words and has_sr:
        word_both += 1
    if w in benchmark_words and has_sr and w not in common_benchmark_words:
        word_onlysem += 1
    if not has_sr and w in common_benchmark_words:
        word_onlydef += 1

# for w in common_benchmark_words:
#     has_sr = syn_anto[w][0] or syn_anto[w][1]
#     if has_sr:
#         word_both += 1
#     if not has_sr:
#         word_onlydef += 1
    # if w in common_benchmark_words and (syn_anto[w][0] or syn_anto[w][1]):
    #     word_both += 1
print(size)
print('benchmark word with both definition and semantic relation:', word_both )
print('benchmark word only in dictionary:', word_onlydef )
print('benchmark word only with semantic relation:', word_onlysem )
with open('./tmp_data/syno_anto.json', 'w') as f:
    json.dump(syn_anto, f, indent=2)






