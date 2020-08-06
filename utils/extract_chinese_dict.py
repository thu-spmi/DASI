# 中文utf8编码范围 u4e00-u9fa5
import re, pprint
import json, pickle
import numpy
# print(0x4e00) # 19968
# print(0x9fa5) # 40869

def extract_xinhua_dict():
    sign = '；。，：‘’“”！、（）'
    from collections import defaultdict
    dic = defaultdict(list)

    def extract(sent):
        sent = re.sub(r'[（〈［].{1,4}[）〉］]', '', sent)
        idx = 0
        while idx < len(sent):
            if 19968 <= ord(sent[idx]) <= 40869 or ord('①') <= ord(sent[idx]) <= ord('⑳'):
                break
            idx += 1
        sent = sent[idx:]
        if '①' in sent:
            # 分割 ①-⑳ 多释义
            for i in range(ord('①'), ord('⑳')):
                sent = sent.replace(chr(i), ' ')
        parts = sent.split()
        res = '。'.join([strip_han(strip_less(s)) for s in parts])
        if '。英' in res:
            res = res[:-1]
        res = res.replace('（）', '')
        return res.strip('。')

    def strip_han(sent):
        res = ''.join([c for c in sent if 19968 <= ord(c) <= 40869 or c in sign])
        return res.strip('。')

    def strip_less(sent):
        if 'a）' in sent:
            idx = sent.index('a）')
            return sent[:idx]
        if '：' in sent:
            idx = sent.index('：')
            return sent[:idx]
        if sent.count('。') > 1:
            return sent[:sent.index('。')]
        else:
            return sent

    with open('../raw_data/chinese_dict.txt', encoding='utf8') as f:
        for line in f:
            if line.strip().startswith('【') and len(line) > 2:
                end = line.index('】')
                w = line[:end + 1][1:-1]
                dic[w].append(extract(line[end + 1:]))

    with open('../tmp_data/xinhua_dict.json', 'w', encoding='utf8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)
        print('total', len(dic))  # 62500


def extract_syn():
    with open('../tmp_data/chinese_dict.json', 'r', encoding='utf8') as f:
        vocab_dict = json.load(f)
    def in_vocab(words):
        return [w for w in words if w in vocab_dict]

    syn = []
    with open('../raw_data/dict_synonym.txt', encoding='utf8') as f:
        for line in f:
            ws = line.split()[1:]
            nws = in_vocab(ws)
            if len(nws) >= 2:
                syn.append(nws)

    with open('../tmp_data/chinese_syn.json', 'w', encoding='utf8') as f:
        json.dump(syn, f, ensure_ascii=False, indent=2)
        print('total', len(syn))  # 17817


def extract_ant():
    ant = []
    with open('../tmp_data/chinese_dict.json', 'r', encoding='utf8') as f:
        vocab_dict = json.load(f)
    def is_in_vocab(words):
        for w in words:
            if w not in vocab_dict:
                return False
        return True
    with open('../raw_data/dict_antonym.txt', encoding='utf8') as f:
        for line in f:
            line = re.sub(r'[—-]', ' ', line)
            ws = line.split()
            if is_in_vocab(ws):
                ant.append(ws)
    with open('../tmp_data/chinese_ant.json', 'w', encoding='utf8') as f:
        json.dump(ant, f, ensure_ascii=False, indent=2)
        print('total', len(ant))  # 18797


def extract_embs():
    sign = '；。，：‘’“”！、（）【】-《》？’'
    embs = dict()
    with open('../tmp_data/chinese_dict.json', 'r', encoding='utf8') as f:
        vocab_dict = json.load(f)
    possible_words = vocab_dict.keys()

    def is_han(w):
        for c in w:
            if 19968 <= ord(c) <= 40869 or c in sign:
                continue
            else:
                return False
        return True

    with open('../raw_data/sgns.merge.char', 'r', encoding='utf8') as f:
        f.readline()
        for i, line in enumerate(f):
            ws = line.split()
            word = ws[0]
            if is_han(word):
                if i < 12000 or word in possible_words:
                    embs[word] = numpy.fromstring('|'.join(ws[1:]),
                                                  dtype=numpy.float32,
                                                  sep='|')
    print(len(embs))
    print(embs.keys())
    with open('../tmp_data/embeddings/ch_w2v_embs.pkl', 'wb') as f:
        pickle.dump(embs, f)


def extract_chinese_dict():
    import jieba
    from collections import Counter
    corpus = []
    with open('../tmp_data/xinhua_dict.json', 'r', encoding='utf8') as f:
        xinhua = json.load(f)
    with open('../raw_data/word_dict.json', 'r', encoding='utf8') as f:
        CMCC_dic = json.load(f)
    for k, v in xinhua.items():
        corpus.append(k)
        corpus.extend(list(jieba.cut('。'.join(v))))
    c = Counter(corpus).most_common(12000)
    dic = {}
    dic['<UNK>'] = len(dic)
    dic['<SEP>'] = len(dic)
    for k in CMCC_dic:
        if k not in dic:
            dic[k] = len(dic)
    for item in c:
        if item[0] not in dic:
            dic[item[0]] = len(dic)
    with open('../tmp_data/chinese_dict.json', 'w', encoding='utf8') as f:
        json.dump(dic, f, ensure_ascii=False)





if __name__ == '__main__':
    # extract_embs()
    extract_syn()
    extract_ant()
    # # 预处理 xinhua.dict
    # with open('../tmp_data/xinhua_dict.json', 'r', encoding='utf8') as f:
    #     xinhua = json.load(f)
    # new_xinhua = {}
    # max_def_len = 0
    # import jieba
    # for k, v in xinhua.items():
    #     defs = []
    #     for s in v:
    #         defs.extend(list(jieba.cut(s)))
    #         defs.append('<SEP>')
    #     defs.pop()
    #     new_xinhua[k] = defs
    #     max_def_len = max(max_def_len, len(defs))
    # print(max_def_len)
    # with open('../tmp_data/xinhua_dict_afterseg.json', 'w', encoding='utf8') as f:
    #     json.dump(new_xinhua, f, indent=2, ensure_ascii=False)










