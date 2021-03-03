import os
import numpy as np
import pandas as pd
import datetime
from bert_serving.client import BertClient

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

text_path = './text'
vector_path = './csv'
vector_file_name = f'total_vector_{current_time}.csv'

bc = BertClient()

text_dirs = os.listdir(text_path)
if not os.path.exists(vector_path):
    os.mkdir(vector_path)

res_list = []
for text_num_file in text_dirs:
    text_num_path = os.path.join(text_path, text_num_file)
    for jidu_file in os.listdir(text_num_path):
        jidu_file_path = os.path.join(text_num_path, jidu_file)
        for text_file in os.listdir(jidu_file_path):
            file_name, extent = os.path.splitext(text_file)
            with open(os.path.join(jidu_file_path, text_file), encoding='utf-8') as f:
                text = f.readlines()
                text = ''.join(text).replace('\n', '').split('。')
                text = list(filter(None, text))
                res_list.append([file_name, text])

print('total num: %d' % len(res_list))

res_vector = []
for person in res_list:
    word_vector = np.zeros(shape=768)
    name, text = person[0], person[1]
    print('start: %s' % name)
    if not text:
        print('文件为空')
        continue
    vectors = bc.encode(text)
    print(vectors.shape)
    for vector in vectors:
        word_vector += vector
    res_vector.append(np.append(name, word_vector / len(vectors)))
    print('已处理: %d' % len(res_vector))

res_vector = pd.DataFrame(res_vector)
res_vector.to_csv(os.path.join(vector_path, vector_file_name), index=False)









