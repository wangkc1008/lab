"""
created by PyCharm
date: 2020/11/28
time: 13:19
user: wkc
"""

import os
import jieba
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import word2vec
import numpy as np
import pandas as pd

data_dir = './data'
data_cut_dir = './data_cut'
model_dir = './model'
model_name = 'word2vec.model'
stop_words_file = './stop_words.txt'
total_text_file = './total_text.txt'
extra_text_file = './extra_text.txt'
result_csv_file = './result.csv'

model_path = os.path.join(model_dir, model_name)


def get_stop_words():
    stop_words = []
    with open(stop_words_file, encoding='utf-8') as f:
        line = f.readline()  # 读取每一行
        while line:
            stop_words.append(line[:-1])  # 读到换行符
            line = f.readline()  # 继续读每一行
    stop_words = set(stop_words)  # 转为set格式，可以去掉一些重复值
    print('停用词读取完毕，共(%d)个词' % len(stop_words))
    return stop_words


def get_all_files(data_dir):
    txt_files = []
    for dir_path, dir_names, file_names in os.walk(data_dir):
        for file_name in file_names:
            if file_name.endswith('.txt'):
                full_name = os.path.join(dir_path, file_name)
                txt_files.append(full_name)
    return txt_files


def handle_text(all_files, stop_words_list):
    total_text = []
    for file in all_files:
        text_file = []
        file_name = os.path.split(file)[-1]
        # file_name = os.path.splitext(os.path.split(file)[-1])[0]

        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            text_cut = list(jieba.cut(text))
            for word in text_cut:
                word = word.strip()
                if not word:
                    continue

                if word not in stop_words_list:
                    total_text.append(word)
                    text_file.append(word)

            if not os.path.exists(data_cut_dir):
                os.mkdir(data_cut_dir)
            with open(os.path.join(data_cut_dir, file_name), 'w', encoding="utf-8") as f2:
                f2.write(' '.join(text_file))

    print('总词个数: %d' % len(total_text))

    with open(extra_text_file, 'r', encoding='utf-8') as f:
        extra_text = f.read()
        if extra_text is not None:
            extra_text_cut = list(jieba.cut(extra_text))
            for word in extra_text_cut:
                if word not in stop_words_list:
                    total_text.append(word)

    print('添加额外词后总词个数: %d' % len(total_text))
    with open(total_text_file, 'w', encoding='utf-8') as f:
        f.write(' '.join(total_text))


def train_model():
    sentences = word2vec.LineSentence(total_text_file)
    # 训练语料
    model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=10, size=200)

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model.save(model_path)
    print('Model保存成功: %s' % model_path)
    

def handle_data(all_files):
    model = word2vec.Word2Vec.load(model_path)
    result = []
    for file in all_files:
        file_name = os.path.splitext(os.path.split(file)[-1])[0]
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            res = np.zeros(shape=200)
            text = text.split(' ')
            for word in text:
                res += model.wv.word_vec(word)
            result.append(np.append(file_name, res/len(text)))
    result = pd.DataFrame(result)
    print(result.shape)
    result.to_csv(result_csv_file, index=False)


if __name__ == '__main__':
    stop_words_list = get_stop_words()
    all_files = get_all_files(data_dir)
    handle_text(all_files, stop_words_list)
    train_model()
    all_cut_files = get_all_files(data_cut_dir)
    handle_data(all_cut_files)
