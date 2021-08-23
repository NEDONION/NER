import tensorflow as tf
import numpy as np
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding


## Session configuration（会话设置）
"""
#设置系统环境变量
#os.environ[‘环境变量名称’]=‘环境变量值’
"""
# 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'，可以是多个，按顺序排列优先级
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 
# default: 0   设置log输出信息的，也就是程序运行时系统打印的信息。0全部输出，log信息共有四个等级，按重要性递增为：INFO（通知）<WARNING（警告）<ERROR（错误）<FATAL（致命的）;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# tf.ConfigProto 用在创建 session 的时候，对 session 进行参数配置，GPU 运算或者 CPU 运算。。
config = tf.ConfigProto()
# 如果为 true，则分配器不会预先分配整个指定的 GPU 内存区域，而是从小开始并根据需要增长。
config.gpu_options.allow_growth = True
# 介于 0 和 1 之间的值，表示为每个进程预分配的可用 GPU 显存的哪一部分。1 表示预分配所有 GPU 显存，0.5 表示该进程分配 ~50% 的可用 GPU 显存。
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory


## hyperparameters（超参数）
"""
#创建解析
#添加参数 add_argument方法
#ArgumentParser.add_argument(name or flags…[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
#name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
#default - 当参数未在命令行中出现时使用的值。
#help - 一个此选项作用的简单描述。
#dest - 被添加到 parse_args() 所返回对象上的属性名。
#（其余参数设定学习https://blog.csdn.net/leo_95/article/details/93783706）
"""
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')#训练数据源
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')#测试数据源
parser.add_argument('--batch_size', type=int, default=32, help='#sample of each minibatch')#每批处理的样本数，越小越费时
#原本的参数是default=40，表示训练模型时的循环次数，但是循环一次需要1个小时。
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')     #迭代次数
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')#隐藏状态的dim
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')#优化器，最佳SGD+Momentum
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')  #如果True，在顶层使用CRF。如果为false，请使用softmax default=True
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')#学习速率
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')#梯度下降量
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')#
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')#训练期间更新嵌入
parser.add_argument('--pretrain_embedding', type=str, default='rando', help='use pretrained char embedding or init it randomly')#使用预处理的 char embedding或者随机初始化
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')#随机初始化字符嵌入
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')#在每次迭代之前清洗训练数据
parser.add_argument('--mode', type=str, default='demo', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1571752775', help='model for test and demo')
args = parser.parse_args()


## get char embeddings(获取字符嵌入)
"""
#os.path.join()函数用于路径拼接文件路径，可以传入多个路径
#read_dictionary 出自 data.py
"""
# word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))
word2id = read_dictionary()
#测试代码
#print('执行read_dictionary():',type(word2id),word2id)  # <class 'dict'> 
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)#3095*300维字典
#    测试代码
    #print('执行random_embedding():',type(embeddings),len(embeddings))# <class 'numpy.ndarray'>
    #后加代码：保存到pretrain_embedding.npy
    #np.save("pretrain_embedding.npy", embeddings)
    
else:
    embedding_path = './pretrain_embedding.npy'        # ？？？？没有这个文件
    embeddings = np.array(np.load(embedding_path), dtype='float32')
    #后加代码：保存到pretrain_embedding.npy
    np.save("pretrain_embedding.npy", embeddings)

## read corpus and get training data（默认演示（demo）指令，如果输入的不是演示的指令，就读取语料库加载训练数据）
if args.mode != 'demo':
    train_path = os.path.join('.', args.train_data, 'train_data')
    test_path = os.path.join('.', args.test_data, 'test_data')
    train_data = read_corpus(train_path)   #2220538行
    test_data = read_corpus(test_path); test_size = len(test_data)   #177233行
#测试代码
   # print('加载训练数据：',type(train_data),train_data)   #<class 'list' [([‘当’，‘’，‘’]，['O','','']),([],[])]> 每个（）中的内容以'\n'分割，在train_data中是空一行。训练数据46364句
#    print('加载测试数据：',type(test_data),test_data,test_size)   #测试数据有4631句

## paths setting（地址设置）
paths = {}
#如果不训练新的模型，就默认执行原有模型，文件路径加时间标记
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
output_path = os.path.join('.', args.train_data+"_save", timestamp)
"""
#os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。
#os.makedirs()用于创建文件路径
"""
if not os.path.exists(output_path): os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)

#'''打印输出日志'''
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))


## training model
#BiLSTM_CRF(默认参数，3096*300维字典，标签（data.py），反序列wordid，地址列表，gpu设置)
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)

    ## train model on the whole training data
    print("train data: {}".format(len(train_data)))#一种格式化字符串的函数 str.format()，它增强了字符串格式化的功能。基本语法是通过 {} 和 : 来代替以前的 % 。
   #train data: 50658
    model.train(train=train_data, dev=test_data)  # use test_data as the dev_data to see overfitting phenomena


## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path) ##会自动找到最近保存的变量文件
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()#构建系统的graph
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)#会自动找到最近保存的变量文件
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()#构建系统的graph
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        print(ckpt_file)
        saver.restore(sess, ckpt_file)
        if True:
            print('Please input your sentence:') #输入要测试的中文内容
            #demo_sent = input()
            demo_sent = '特朗普上台后,不断强化美台实质关系,蔡英文当局向美国的"印太战略"积极靠拢.'
            if demo_sent == '' or demo_sent.isspace():  #str.isspace 检测字符串是否只包含空格，是则返回 True，否则返回 False。
                print('See you next time!')
                #break
            else:
                demo_sent = list(demo_sent.strip())#str.strip（）函数用于删除指定字符串头尾信息（默认为空格）
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                #测试代码
                print('执行model.demo，demo_data=',type(demo_data),demo_data)
                print('+===正在识别中===+')
                tag = model.demo_one(sess, demo_data)
                #测试代码
                print('执行model.demo_one，tag=',type(tag),tag)
                PER, LOC, ORG, EQU= get_entity(tag, demo_sent)
                print('PER: {}\nLOC: {}\nORG: {}\nEQU: {}'.format(PER, LOC, ORG, EQU))
