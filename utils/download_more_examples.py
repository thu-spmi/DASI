'''
从有道词典下载更多例句
'''
from urllib import request
import json, re
import requests
from lxml import etree
import time

class Youdao:
    def __init__(self):
        self.url_1 = 'http://fanyi.youdao.com/openapi.do'
        self.key = '993123434'  #有道API key
        self.keyfrom = 'pdblog' #有道keyfrom

        self.headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    # "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36",
    "Connection": "keep-alive",
    "referer": "http://dict.youdao.com/"
}
        self.url_2 = 'http://dict.youdao.com/w/eng/****/#keyfrom=dict2.index'

    def get_url_1(self, word):
        url = self.url_1 + '?keyfrom=' + self.keyfrom + '&key='+self.key + '&type=data&doctype=json&version=1.1&q=' + word
        result = request.urlopen(url).read().decode('utf-8')
        self.res = json.loads(result)
        cadidates = []
        try:
            for dic in self.res['web']:
                if word in dic['key']:
                    s = dic['key'].replace(word, '****').lower()
                    s = s.replace('****', word)
                    cadidates.append(s)
        except:
            cadidates = []
        return cadidates

    def get_url_2(self, word):
        url = self.url_2.replace('****', word)
        response = requests.get(url=url, headers=self.headers)
        selector = etree.HTML(response.text)
        cadidates = []
        try:
            for n in ['1','2','3']:
                content = selector.xpath('//*[@id="bilingual"]/ul/li[%s]/p[1]'%n)
                if content:
                    sentence = []
                    for i in content[0]:
                        sentence.append(i.xpath('string(.)'))
                    if word in ' '.join(sentence):
                        s = ' '.join(sentence).replace(word, '****')
                        s = re.sub('t[ ’\']+s', 'ts', s)
                        s = re.sub('["…‘“”,.:?]', ' ', s)
                        s = ' '.join(s.lower().split()).replace('****', word)
                        cadidates.append(s)
        except:
            cadidates = []
        return cadidates


if __name__ == '__main__':
    youdao = Youdao()
    with open('./tmp_data/word_dict.json', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    with open('./tmp_data/wordnet', encoding='utf-8') as f:
        wordnet = json.load(f)
    for w in vocab_dict:
        if w in wordnet and len(wordnet[w]["examples"])<=3:
            print(w)
            downloads = youdao.get_url_1(w)+youdao.get_url_2(w)
            examples = []
            try:
                for sent in downloads:
                    sent += ' <SEP>'
                    if len(sent.split())>2:
                        idx = sent.split().index(w)
                        examples.append([sent, idx])
                wordnet[w]["examples"].extend(examples)
            except:
                examples = []
            print(examples)
            time.sleep(3)
    with open('./tmp_data/wordnet', 'w', encoding='utf-8') as f:
        json.dump(wordnet, f, indent=2)


