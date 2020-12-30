"""
created by PyCharm
date: 2020/11/28
time: 12:16
user: wkc
"""
import jieba
import jieba.analyse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import word2vec
#文件位置需要改为自己的存放路径
#将文本分词
with open('./practice_jieba.txt',encoding='utf-8') as f:
    document = f.read()
    document_cut = jieba.cut(document)
    result = ' '.join(document_cut)
    with open('./result_jieba.txt', 'w',encoding="utf-8") as f2:
        f2.write(result)
#加载语料
sentences = word2vec.LineSentence('./result_jieba.txt')
#训练语料
path = get_tmpfile("word2vec.model") #创建临时文件
model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=10,size=200)
# model.save("word2vec.model")
# model = Word2Vec.load("word2vec.model")
#输入与“贪污”相近的100个词
res = model.wv.word_vec('对角')
print(res)
print(len(res))
print(sum(res) / 200)
