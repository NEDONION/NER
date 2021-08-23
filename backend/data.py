import sys, pickle, os, random
import numpy as np

## tags, BIO
tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6,
             "B-EQU": 7, "I-EQU": 8 
             }


def read_corpus(corpus_path):
    """
    read corpus and return the list of samples（读取训练语料库（2220538行）和测试预料库（177233行））
    :param corpus_path:
    :return: data
    """
    data = []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []     #sent_是汉字，tag_是标签
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data


def vocab_build(vocab_path, corpus_path, min_count):
    """

    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence2id(sent, word2id):
    """

    :param sent:
    :param word2id:
    :return:
    """
    sentence_id = []
    for word in sent:
        if word.isdigit():#isdigit函数检测字符串中是否只包含数字字符。若全部是由数字字符组成的字符串，则返回True，否则返回False。
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
   
    return sentence_id


# def read_dictionary(vocab_path):
def read_dictionary():
    """

    :param vocab_path:
    :return:
    """
    
#    序列化方法pickle.dumps()
#    反序列化方法pickle.load()

    # vocab_path = os.path.join(vocab_path)
    with open("./data_path/word2id.pkl", 'rb') as fr:
        word2id = pickle.load(fr)
    # print('vocab_size:', len(word2id))  #vocab_size: 3905()-------word2id.pkl的长度
   #测试代码
    #print("执行read_dictionary(),word2id=",word2id) 
    return word2id


def random_embedding(vocab, embedding_dim):
    """

    :param vocab:
    :param embedding_dim:
    :return:
    """
    #从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))  #维度为（3095，300）
    embedding_mat = np.float32(embedding_mat)  #转换为32位浮点数，也可以是64位，这样精度更高
   #测试代码
   # print(embedding_mat)
    return embedding_mat

#返回句子的列表和句子的长度列表
def pad_sequences(sequences, pad_mark=0):
    """

    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences)) #得到sequences的最大值，就是句子的最大长度
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    return seq_list, seq_len_list


def batch_yield(data, batch_size, vocab, tag2label, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param vocab:
    :param tag2label:
    :param shuffle:
    :return:
    """
    if shuffle:
        random.shuffle(data) #打乱顺序

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence2id(sent_, vocab)#对一个句子中所有词的词向量进行加权平均，每个词向量的权重可以表示为a/（a+p(w)），其中a为参数，p(w)为词w的频率。
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:            
            yield seqs, labels   
            #yield 是一个类似 return 的关键字，迭代一次遇到yield时就返回yield后面(右边)的值。
            #只要出现了yield表达式（Yield expression），那么事实上定义的是一个generator function， 调用这个generator function返回值是一个generator。
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)
        
    #测试代码
#    print('执行batch_yield(),seqs',seqs,type(seqs))
#    print('执行batch_yield(),seqs',labels,type(labels))
    

    if len(seqs) != 0:
        yield seqs, labels

