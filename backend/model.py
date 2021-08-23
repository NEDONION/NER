import numpy as np
import os, time, sys
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from data import pad_sequences, batch_yield
from utils import get_logger
from eval import conlleval


class BiLSTM_CRF(object):
#    ''''初始化模型参数设置'''
    def __init__(self, args, embeddings, tag2label, vocab, paths, config):
        self.batch_size = args.batch_size #每批处理的样本数default=64
        self.epoch_num = args.epoch      #训练的迭代数default=40
        self.hidden_dim = args.hidden_dim  #隐藏状态的dim ，default=300
        self.embeddings = embeddings   #3095*300
        self.CRF = args.CRF     #如果True，在顶层使用CRF。如果为false，请使用softmax default=True
        self.update_embedding = args.update_embedding
        self.dropout_keep_prob = args.dropout #default=0.5
        self.optimizer = args.optimizer   #优化器default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD'
        self.lr = args.lr  #学习速率default=0.001,
        self.clip_grad = args.clip  #渐变剪裁量，default=5.0
        self.tag2label = tag2label  #标签
        self.num_tags = len(tag2label) #（0-6）
        self.vocab = vocab     # 3095*300
        self.shuffle = args.shuffle  #每次迭代之前清洗数据，default=True
        self.model_path = paths['model_path']#模型地址
        self.summary_path = paths['summary_path']#总结地址
        self.logger = get_logger(paths['log_path'])#日志地址
        self.result_path = paths['result_path']#结果地址
        self.config = config #设置会话，gpu使用

    def build_graph(self):
        
#         Tensorflow的设计理念称之为计算流图,在编写程序时，首先构筑整个系统的graph
        self.add_placeholders()
        self.lookup_layer_op()
        self.biLSTM_layer_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

#        placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
#          tf.placeholder( dtype,shape=None,name=None)
#               dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
#               shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
#               name：名称      

  
    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")#序列长度
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
        
     
#        variable_scope通过变量名获取变量。
#        变量可以通过tf.Varivale来创建，tf.Variable的变量名是一个可选项，通过name=’v’的形式给出。但是tf.get_variable必须指定变量名。
#
#        tf.nn.embedding_lookup()函数主要是选取一个张量里面索引对应的元素，根据ids选取params中的元素。
#        params: 表示完整的嵌入张量，或者除了第一维度之外具有相同形状的P个张量的列表，表示经分割的嵌入张量
#        ids: 一个类型为int32或int64的Tensor，包含要在params中查找的id
#        name: 操作名称（可选）
#
#        tf.nn.dropout()是tensorflow里面为了防止或减轻过拟合而使用的函数，它一般用在全连接层
#        Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，这次训练过程中不更新权值，也不参加神经网络的计算。
#        但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了            

#模型的第一层是look-up层，利用预训练或随机初始化的embedding矩阵将句子中的每个字x_i由one-hot向量映射为低维稠密的字向量（character embedding）x_i\in R^2，d是embedding的维度。在输入下一层之前，设置dropout以缓解过拟合。
    def lookup_layer_op(self):
             
        with tf.variable_scope("words"):
            _word_embeddings = tf.Variable(self.embeddings,
                                           dtype=tf.float32,
                                           trainable=self.update_embedding,
                                           name="_word_embeddings")
            
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings,
                                                     ids=self.word_ids,
                                                     name="word_embeddings")
          
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout_pl)
        #测试代码
        #print('执行lookup_layer_op(),self.word_embeddings=',self.word_embeddings)#self.word_embeddings= Tensor("dropout_1/mul_1:0", shape=(?, ?, 300), dtype=float32)
        
     
# Class tf.contrib.rnn.LSTMCell继承自：LayerRNNCell
# LSTMCell(LSTM cell中的单元数量，即隐藏层神经元数量,默认300)

#tf.get_variable函数来创建或者获取变量
#tf.get_variable(name,  shape, initializer): name就是变量的名称，shape是变量的维度，initializer是变量初始化的方式
#
#tf.shape将矩阵的维度输出成一个维度的矩阵，如[m*n*t]，就是查询矩阵的维度
#tf.shape(input：输入张量或稀疏张量；name：命名；out_type：默认tf.int32类型；)
#tf.reshape将tensor变换为参数shape的形式。
#tf.reshape（tensor：输入张量；shape：列表形式，可以存在-1； -1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1；name：命名；）
# tf.matmul（） 矩阵乘法


    def biLSTM_layer_op(self):
        with tf.variable_scope("bi-lstm"):                       
            cell_fw = LSTMCell(self.hidden_dim)###前项隐藏层
            cell_bw = LSTMCell(self.hidden_dim)#后项隐藏层

            (output_fw_seq, output_bw_seq), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,# 前向RNN
                cell_bw=cell_bw, # 后向RNN
                inputs=self.word_embeddings, # 输入
                sequence_length=self.sequence_lengths, # 输入序列的实际长度（可选，默认为输入序列的最大长度）
                dtype=tf.float32)  # 初始化和输出的数据类型（可选）
            #tensorflow中用来拼接张量的函数tf.concat()
            output = tf.concat([output_fw_seq, output_bw_seq], axis=-1) 
            # 是一个包含前向cell输出tensor和后向cell输出tensor组成的元组。time_major = False,所以输入为[batch_size,time_step,embedding_dim],所以这样连接,相当于 axis = 2
            output = tf.nn.dropout(output, self.dropout_pl)#防止过拟合,output= Tensor("bi-lstm/dropout/mul_1:0", shape=(?, ?, 600), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'>
            #测试代码
            #print('执行biLSTM_layer_op(),output=',output,type(output))#self.logits= Tensor("proj/Reshape_1:0", shape=(?, ?, 7), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'>

        with tf.variable_scope("proj"):
            W = tf.get_variable(name="W",
                                shape=[2 * self.hidden_dim, self.num_tags],
                                initializer=tf.contrib.layers.xavier_initializer(),#该函数返回一个用于初始化权重的初始化程序 “Xavier” 。这个初始化器是用来保持每一层的梯度大小都差不多相同。W= <tf.Variable 'proj/W:0' shape=(600, 7) dtype=float32_ref> <class 'tensorflow.python.ops.variables.RefVariable'>
                                dtype=tf.float32)
            #测试代码
           # print('执行biLSTM_layer_op(),W=',W,type(W))
            
            b = tf.get_variable(name="b",
                                shape=[self.num_tags],
                                initializer=tf.zeros_initializer(),#初始化全部为零b= <tf.Variable 'proj/b:0' shape=(7,) dtype=float32_ref> <class 'tensorflow.python.ops.variables.RefVariable'>初始化0矩阵
                                dtype=tf.float32)
            #测试代码
           # print('执行biLSTM_layer_op(),b=',b,type(b))
            s = tf.shape(output)#s= Tensor("proj/Shape:0", shape=(3,), dtype=int32) <class 'tensorflow.python.framework.ops.Tensor'>
            #测试代码
           # print('执行biLSTM_layer_op(),s=',s,type(s))
            output = tf.reshape(output, [-1, 2*self.hidden_dim]) #output= Tensor("proj/Reshape:0", shape=(?, 600), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'>
            #测试代码
           # print('执行biLSTM_layer_op(),output=',output,type(output))
            pred = tf.matmul(output, W) + b  # pred =Tensor("proj/add:0", shape=(?, 7), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'>

            #测试代码
          #  print('执行biLSTM_layer_op(),pred=',pred,type(pred))
            self.logits = tf.reshape(pred, [-1, s[1], self.num_tags])
             #测试代码
           # print('执行biLSTM_layer_op(),self.logits=',self.logits,type(self.logits))#self.logits= Tensor("proj/Reshape_1:0", shape=(?, ?, 7), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'>

            


#crf_log_likelihood（）最大似然估计损失函数
#inputs: 一个形状为[batch_size, max_seq_len, num_tags] 的tensor,一般使用BILSTM处理之后输出转换为他要求的形状作为CRF层的输入
#tag_indices: 一个形状为[batch_size, max_seq_len] 的矩阵,其实就是真实标签
##sequence_lengths: 一个形状为 [batch_size] 的向量,表示每个序列的长度
#
#返回：
#log_likelihood: 包含给定序列标签索引的对数似然的标量
#transition_params: 形状为[num_tags, num_tags] 的转移矩阵

#tf.reduce_mean（） 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度，如果不指定，则计算所有元素的均值）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
#它求得是张量最后一维与标签的交叉熵，再对最后一维求和。

    def loss_op(self):
        if self.CRF:
            log_likelihood, self.transition_params = crf_log_likelihood(inputs=self.logits,
                                                                   tag_indices=self.labels,
                                                                   sequence_lengths=self.sequence_lengths)
            self.loss = -tf.reduce_mean(log_likelihood) #Tensor("Neg:0", shape=(), dtype=float32) <class 'tensorflow.python.framework.ops.Tensor'>
            #测试代码
          #  print('执行loss_op(),self.loss=',self.loss,type(self.loss))
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                    labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths) #TensorFlow张量变换函数，每一维前面为True，后面为False
            losses = tf.boolean_mask(losses, mask) #将mask为True的地方保存下来。
            self.loss = tf.reduce_mean(losses)#平均值
             #测试代码
           # print('执行loss_op(),self.loss=',self.logits,type(self.loss))

        tf.summary.scalar("loss", self.loss)#用于收集一维标量，返回一个字符串类型的标量张量

    def softmax_pred_op(self):
        if not self.CRF:
            self.labels_softmax_ = tf.argmax(self.logits, axis=-1)#返回沿着某个维度最大值的位置    
            self.labels_softmax_ = tf.cast(self.labels_softmax_, tf.int32)#tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换
           #self.labels_softmax_= Tensor("Cast:0", shape=(?, ?), dtype=int32) <class 'tensorflow.python.framework.ops.Tensor'>
            #测试代码
            #print('执行softmax_pred_op,self.labels_softmax_=',self.labels_softmax_,type(self.labels_softmax_))

#训练优化，可以选择不同的优化算法
    def trainstep_op(self):
        with tf.variable_scope("train_step"):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)#tf.train.get_global_step() 方法返回的是的 global_step作为name的tensor,  tensor参数与global_step = tf.Variable(0, name=“global_step”, trainable=False) 完全相同。
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl) #Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)

            grads_and_vars = optim.compute_gradients(self.loss)#计算loss中可训练的var_list中的梯度。返回(gradient, variable)对的list。
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)#返回回一个执行梯度更新的ops。
            #测试代码
          # print('执行trainstep_op(),self.train_op=',self.train_op,type(self.train_op))
   
    def init_op(self):
        self.init_op = tf.global_variables_initializer()  #tf.global_variables_initializer() 添加节点用于初始化所有的变量。返回一个初始化所有全局变量的操作（Op）
   #测试代码
       # print('执行init_op,self.init_op =',self.init_op ,type(self.init_op ))

    def add_summary(self, sess):
        """

        :param sess:
        :return:
        """
        self.merged = tf.summary.merge_all() #自动管理模式下，导入已保存的模型继续训练
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)#指定一个文件用来保存图。以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中
        #self.file_writer = tf.compat.v1.summary.FileWriter(self.summary_path, sess.graph)
    def train(self, train, dev):
        """

        :param train:
        :param dev:
        :return:
        """
  #tf.global_variables()查看全部变量      
  #Saver类提供了向checkpoints文件保存和从checkpoints文件中恢复变量的相关方法。Checkpoints文件是一个二进制文件，它把变量名映射到对应的tensor值      
        saver = tf.train.Saver(tf.global_variables())#   saver= <tensorflow.python.training.saver.Saver object at 0x00000296A659E198> <class 'tensorflow.python.training.saver.Saver'>
        #测试代码
       # print('执行train(),saver=',saver,type(saver)) 
        
        with tf.Session(config=self.config) as sess:   #在Tensorflow中通过tf.Session来完成图的执行操作。通过Session对象，可以执行途中的Operation，同时计算得到Tensor的值。
            sess.run(self.init_op)     #初始化全局变量
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, train, dev, self.tag2label, epoch, saver)
           

    def test(self, test):
        saver = tf.train.Saver()#Saver类可以自动的生成checkpoint文件。这让我们可以在训练过程中保存多个中间结果。
        with tf.Session(config=self.config) as sess:
            self.logger.info('=========== testing ===========')
            saver.restore(sess, self.model_path)#恢复模型
            
            label_list, seq_len_list = self.dev_one_epoch(sess, test)
            #测试代码
            print('输出test中的label_list',label_list)
            print('输出test中的seq_len_list',seq_len_list)
            self.evaluate(label_list, seq_len_list, test)

    def demo_one(self, sess, sent):
        """

        :param sess:
        :param sent: 
        :return:
        """
        label_list = []
        for seqs, labels in batch_yield(sent, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, _ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_)
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label
        tag = [label2tag[label] for label in label_list[0]]
        return tag

    def run_one_epoch(self, sess, train, dev, tag2label, epoch, saver):
        """

        :param sess:
        :param train:
        :param dev:
        :param tag2label:
        :param epoch:
        :param saver:
        :return:
        """
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size    #batch_size=64（批处理数目）

        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  #返回以可读字符串表示的当地时间
        #batch_yield（）来自data.py
        batches = batch_yield(train, self.batch_size, self.vocab, self.tag2label, shuffle=self.shuffle)
      #执行batch_yield(),batches <generator object batch_yield at 0x00000169518F9BA0> <class 'generator'>
        #测试代码
        #print('执行batch_yield(),batches',batches,type(batches))

#python中的enumerate 函数用于遍历序列中的元素以及它们的下标。enumerate 能同时循环索引和元素：索引在前、元素在后
        for step, (seqs, labels) in enumerate(batches):
#sys.stdout.write实现打印刷新，本质功能是print（）
            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            
            feed_dict, _ = self.get_feed_dict(seqs, labels, self.lr, self.dropout_keep_prob)#dropout_keep_prob一般都设置为1，也就是保留全部结果,这里是0.5
            #测试代码
            #print('执行get_feed_dict,feed_dict, =',feed_dict, type(feed_dict),)
            #print('执行get_feed_dict,, _=', _,type(_))
            
            #sess.run(),执行初始化后的所有操作。
            _, loss_train, summary, step_num_ = sess.run([self.train_op, self.loss, self.merged, self.global_step],
                                                         feed_dict=feed_dict)
            #测试代码
#            print('执行sess.run(),_=',type(_), _)
#            print('执行sess.run(),loss_train=',type(loss_train), loss_train)
#            print('执行sess.run(),summary=',type(summary), summary)
#            print('执行sess.run(),step_num_=',type(step_num_), step_num_)
                       
            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:
                self.logger.info(
                    '{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, step + 1,
                                                                                loss_train, step_num))

            self.file_writer.add_summary(summary, step_num)#结果写入文件

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step=step_num)#保存训练模型

        self.logger.info('===========validation / test===========')
        label_list_dev, seq_len_list_dev = self.dev_one_epoch(sess, dev)
        #测试代码
        #print('执行train.dev_one_epoch，label_list_dev=',label_list_dev)
        #print('执行train.dev_one_epoch，seq_len_list_dev=',seq_len_list_dev)
        self.evaluate(label_list_dev, seq_len_list_dev, dev, epoch)

    def get_feed_dict(self, seqs, labels=None, lr=None, dropout=None):
        """

        :param seqs:
        :param labels:
        :param lr:
        :param dropout:
        :return: feed_dict
        """
       # pad_sequences（）来自data.py
        word_ids, seq_len_list = pad_sequences(seqs, pad_mark=0) #dropout_keep_prob一般都设置为1，也就是保留全部结果
        #测试代码
        #print('执行pad_sequences(),word_ids, ' ,word_ids,type(word_ids))
       # print('执行pad_sequences(), seq_len_list' , seq_len_list, type(seq_len_list))

        feed_dict = {self.word_ids: word_ids,
                     self.sequence_lengths: seq_len_list}
        if labels is not None:
            labels_, _ = pad_sequences(labels, pad_mark=0)
            feed_dict[self.labels] = labels_
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout

        return feed_dict, seq_len_list

    def dev_one_epoch(self, sess, dev):
        """

        :param sess:
        :param dev:
        :return:
        """
        label_list, seq_len_list = [], []
        for seqs, labels in batch_yield(dev, self.batch_size, self.vocab, self.tag2label, shuffle=False):
            label_list_, seq_len_list_ = self.predict_one_batch(sess, seqs)
            label_list.extend(label_list_) #extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）。
            seq_len_list.extend(seq_len_list_)
        return label_list, seq_len_list

    def predict_one_batch(self, sess, seqs):
        """

        :param sess:
        :param seqs:
        :return: label_list
                 seq_len_list
        """
        feed_dict, seq_len_list = self.get_feed_dict(seqs, dropout=1.0)#初始化参数

        if self.CRF:
            logits, transition_params = sess.run([self.logits, self.transition_params],#执行运算
                                                 feed_dict=feed_dict)
            label_list = []
            for logit, seq_len in zip(logits, seq_len_list):#zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
                viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)   #解码tensorflow外标签的最高评分序列。
                label_list.append(viterbi_seq)
            return label_list, seq_len_list

        else:
            label_list = sess.run(self.labels_softmax_, feed_dict=feed_dict)
            return label_list, seq_len_list

    def evaluate(self, label_list, seq_len_list, data, epoch=None):
        """

        :param label_list:
        :param seq_len_list:
        :param data:
        :param epoch:
        :return:
        """
        label2tag = {}
        for tag, label in self.tag2label.items():
            label2tag[label] = tag if label != 0 else label

        model_predict = []
        for label_, (sent, tag) in zip(label_list, data):
            tag_ = [label2tag[label__] for label__ in label_]
            sent_res = []
            if  len(label_) != len(sent):
                print(sent)
                print(len(label_))
                print(tag)
            for i in range(len(sent)):
                sent_res.append([sent[i], tag[i], tag_[i]])
            model_predict.append(sent_res)
        epoch_num = str(epoch+1) if epoch != None else 'test'
        label_path = os.path.join(self.result_path, 'label_' + epoch_num)
        metric_path = os.path.join(self.result_path, 'result_metric_' + epoch_num)
        for _ in conlleval(model_predict, label_path, metric_path):
            #self.logger.info(_)
            #测试代码
            print(_)

