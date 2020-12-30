import tensorflow as tf
import jieba
import re
import numpy as np
import math
import pickle as pkl
import collections
import os
import os.path as path
from collections import Counter


class word2vec():
    def __init__(self,
                 vocab_list=None,
                 embedding_size=200,
                 win_len = 3,
                 learning_rate=1.0,
                 num_sampled=1000,
                 logdir='/tmp/simple_word2vec',
                 model_path=None
                 ):

        self.batch_size = None
        if model_path!=None:
            self.load_model(model_path)
        else:
            assert type(vocab_list) ==list   #判断传入的类型是否为list
            self.vocab_list = vocab_list
            self.vocab_size = vocab_list._len_()
            self.win_len = win_len
            self.embedding_size = embedding_size
            self.num_sampled = num_sampled
            self.logdir = logdir


            #对每个词进行id的映射
            self.word2id = {}             #word => id映射
            for i in range(self.vocab_size):
                self.word2id[self.vocab_list[i]]  =i
            #指定训练参数
            self.train_words_num = 0
            self.train_sents_num = 0
            self.train_times_num = 0

            self.train_loss_records = collections.deque(maxlen=10)  #保存最近10次的误差
            self.train_loss_k10 = 0

    #构造计算图模型
        self.build_graph()
        self.init_op()
        if model_path!=None:
            tf_model_path = os.path.join(model_path,'tf_vars')
            self.saver.restore(self.sess,tf_model_path)
    def init_op(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.summary_writer = tf.train.SummaryWriter(self.logdir,self.sess.graph)

    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32,shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32,shape=[self.batch_size,1])
            self.embedding_dict = tf.Variable(tf.random_uniform([self.vocab_size,self.embedding_size],-1.0,1.0))   #初始化
            stddev = 1.0 / math.sqrt(self.embedding_dict)
            self.nec_weight = tf.Variable(tf.truncated_normal([self.vocab_size,self.embedding_size,stddev]))
            self.nec_biases = tf.Variable(tf.zeros([self.vocab_size]))

            embed = tf.nn.embedding_lookup(self.embedding_dict,self.train_inputs)   #在embedding_dict中查找train_inputs
            #损失函数
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=self.nec_weight,
                               biases=self.nec_biases,
                               inputs=embed,
                               labels=self.train_labels,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocab_size))
            #训练
            tf.scalar_summary('loss',self.loss)
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)
            #测试id
            self.test_word_id = tf.placeholder(tf.int32,shape=[None])
            #归一化，先求模
            vec_l2_model = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_dict),1,keep_dims=True))  #求模
            avg_l2_model = tf.reduce_mean(vec_l2_model)
            tf.scalar_summary('avg_vec_model',avg_l2_model)
            self.normed_embedding = self.embedding_dict/vec_l2_model                 #得到归一化向量
            #测试
            test_embed = tf.nn.embedding_lookup(self.embedding_dict,self.test_word_id)
            #相似度计算
            self.similarity = tf.matmul(test_embed,self.normed_embedding,transpose_b=True)
            #全局变量初始化
            self.init = tf.global_variables_initializer()
            self.merged_summary_op = tf.merge_all_summaries
            #模型保存
            self.saver = tf.train.Saver()
    def train_by_sentence(self,input_sentence=[]):
        sent_num = input_sentence.__len__()
        batch_inputs = []
        batch_labels = []
        for sent in input_sentence:
            for i in range(sent.__len__()):
                start = max(0,i-self.win_len)
                end = min(sent.__len__(),i+self.win_len)
                for index in range(start,end):
                    if index == i:
                        continue
                    else:
                        input_id = self.word2id.get(sent[i])
                        label_id = self.word2id.get(sent[index])
                        if not (input_id and label_id):
                            continue
                        batch_inputs.append(input_id)
                        batch_labels.append(label_id)
        if len(batch_inputs)==0:
            return
        batch_inputs = np.array(batch_inputs,dtype=np.int32)
        batch_labels = np.array(batch_labels,dtype=np.int32)
        batch_labels = np.reshape(batch_labels,[batch_labels.__len__(),1])
        feed_dict = {
                     self.train_inputs:batch_inputs,
                     self.train_labels:batch_labels}
        _,loss_val,summary_str = self.sess.run([self.train_op,self.loss,self.merged_summary_op])
        self.train_loss_records.append(loss_val)
        self.train_loss_k10 = np.mean(self.train_loss_records)
        if self.train_sents_num % 1000 ==0:
            self.summary_writer.add_summary(summary_str,self.train_sents_num)
            print("(a) sentences dealed, loss: (b)"
                  .format(a=self.train_sents_num,b=self.train_loss_k10))
        self.train_words_num += batch_inputs.__len__()
        self.train_sentence_num += input_sentence.__len__()
        self.train_times += 1
        self.train_sentence_num += input_sentence.__len__()
        self.train_times_num += 1
    #模型保存
    def save_model(self,save_path):
        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model = {}
        var_names = ['vocab_size',
                     'vocab_list',
                     'learning_rate',
                     'word2id',
                     'embdding_size',
                     'logdir',
                     'win_len',
                     'num_sampled',
                     'train_words_num',
                     'train_sent_num',
                     'train_times_num',
                     'train_loss_records',
                     'train_loss_k10',
                     ]
        for var in var_names:
            model[var] = eval('self.'+var)

        param_path = os.path.join(save_path,'params.pkl')
        if os.path.exists(param_path):
            os.remove(param_path)
        with open(param_path,'wb') as f:
            pkl.dump(model,f)
        #记录模型
        tf_path = os.path.join(save_path,'tf_vars')
        if os.path.exists(tf_path):
            os.remove(tf_path)
        self.saver.save(self.sess,tf_path)
    #模型加载
    def load_model(self,model_path):
        if not os.path.exists(model_path):
            raise RuntimeError('file not exists')
        param_path = os.path.join(model_path,'params.pkl')
        with open(param_path,'rb') as f:
            model = pkl.load(f)
            self.vocab_list = model['vocab_list']
            self.vocab_size = model['vocab_size']
            self.logdir = model['logdir']
            self.word2id = model['word2id']
            self.embedding_size = model['embedding_size']
            self.learning_rate = model['learning_rate']
            self.win_len = model['win_len']
            self.num_sampled = model['num_sampled']
            self.train_words_num = model['train_words_num']
            self.train_sent_num = model['train_sent_num']
            self.train_times_num = model['train_times_num']
            self.train_loss_records = model['train_loss_records']
            self.train_loss_k10 = model['train_loss_k10']

    def cal_similarity(self,test_word_id):
        sim_matrix = self.sess.run(self.similarity,feed_dict = {self.test_word_id:test_word_id})
        test_words = []
        near_words = []
        for i in range(test_word_id._len_()):
            test_words.append(self.vocab_list[test_word_id][i])
            nearest_id = [sim_matrix[i,:].argsort()[1:10]]
            nearest_word = [self.vocab_list[x] for x in nearest_id]
            near_words.append(nearest_word)
        return test_words,near_words
if __name__ == '__main__':
    stop_words = []  # 停用词
    with open('stop_words.txt', encoding='utf-8') as f:
        line = f.readline()  # 读取每一行
        while line:
            stop_words.append(line[:-1])  # 读到换行符
            line = f.readline()  # 继续读每一行
    stop_word = set(stop_words)  # 转为set格式，可以去掉一些重复值
    print('停用词读取完毕，共(n)个单词'.format(n=len(stop_words)))
    raw_word_list = []  # 分词后词语
    sentence_list = []
    with open('practice_jieba.txt', encoding='gbk') as f:
        line = f.readline()  # 读取每一行
        while line:
            while '\n' in line:
                line = line.replace('\n', '')  # 用空格代替换行符
            while ' ' in line:
                line = line.replace(' ','')
            if len(line) > 0:
                raw_words = list(jieba.cut(line,cut_all=False))  # 将分词完的结果传入list
                dealed_words = []
                for word in raw_words:
                    if word not in stop_words:
                        raw_word_list.append(word)
                        dealed_words.append(word)
                sentence_list.append(dealed_words)    #将分好的词放入sentence_list
            line = f.readline()                           #继续读line
    word_count = collections.Counter(raw_word_list)   #统计词频
    print('文本中总共有(n1)个单词，不重复单词数(n2),选取前30000个单词进入词典'
          .format(n1=len(raw_word_list),n2=len(word_count)))
    word_count = word_count.most_common(3000)         #指定排序阈值
    word_list = [x[0] for x in word_count]            #得到所有的词
    w2v = word2vec(vocab_list = word_list,
                   embedding_size = 200,
                   learning_rate = 1,
                   num_sampled = 10,
                   win_len=2,
                   logdir='/tmp/280')                #tensorboard记录地址
    num_steps = 10000
    for i in range(num_steps):
        sent = sentence_list[i]
        w2v.train_by_sentence([sent])
    w2v.save_model('model')
    w2v.load_model('model')
    test_word = ['天地','级别']
    test_id = [word_list.index(x) for x in test_word]  #将test_word转为test_id
    test_words,near_words,sim_mean,sim_var = w2v.cal_similarity(test_id)
    print(test_words,near_words,sim_mean,sim_var)