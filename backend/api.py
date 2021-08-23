import tensorflow as tf
import os
from data import random_embedding, tag2label
from flask import Flask, jsonify, request
from flask_cors import CORS
from main import word2id
from argparse import Namespace
from model import BiLSTM_CRF
from utils import get_entity

import json
# 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'，可以是多个，按顺序排列优先级
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# default: 0   设置log输出信息的，也就是程序运行时系统打印的信息。0全部输出，log信息共有四个等级，按重要性递增为：INFO（通知）<WARNING（警告）<ERROR（错误）<FATAL（致命的）;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.ConfigProto 用在创建 session 的时候，对 session 进行参数配置，GPU 运算或者 CPU 运算。。
config = tf.ConfigProto()
# 如果为 true，则分配器不会预先分配整个指定的 GPU 内存区域，而是从小开始并根据需要增长。
config.gpu_options.allow_growth = True
# 介于 0 和 1 之间的值，表示为每个进程预分配的可用 GPU 显存的哪一部分。1 表示预分配所有 GPU 显存，0.5 表示该进程分配 ~50% 的可用 GPU 显存。
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory
embeddings = random_embedding(word2id, 300)


paths = {}
# 如果不训练新的模型，就默认执行原有模型，文件路径加时间标记
timestamp = "1571752775"
output_path = os.path.join('.', "data_path_save", timestamp)
print(output_path)
if not os.path.exists(output_path):
    os.makedirs(output_path)
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path):
    os.makedirs(summary_path)
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path):
    os.makedirs(model_path)
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)


log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
ckpt_file = tf.train.latest_checkpoint(model_path)  # 会自动找到最近保存的变量文件
paths['model_path'] = ckpt_file

app = Flask(__name__)

# 跨域支持
def after_request(response):
    # JS前端跨域支持
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
 
app.after_request(after_request)


# @app.route('/', methods=['POST'])
@app.route('/')
def index():
    return jsonify({"code":200, "value": "服务启动成功"})


@app.route('/extract')
def entity():
    text = request.args['content']
    # text = "特朗普和美国州长电话会议录音 怒斥各州州长"
    # print("test:::::::::", request.json.get('content'))
    """
        var data= {
                    data: JSON.stringify({
                        'text': "特朗普和美国州长电话会议录音 怒斥各州州长",
                    }),
                }
                $.ajax({
                url:'http://localhost:5000/',
                type:'POST',
                data:data,
                dataType: 'json',
                success:function(res){
                    console.log(res)
                    console.log(0)

                },
                error:function (res) {
                    console.log(res);
                    console.log(1)
                }
    :return:
    """
    # data = json.loads(request.form.get('data'))
    # text = data['text']
    
    args = Namespace(CRF=True, batch_size=32, clip=5.0, demo_model='1571752775', dropout=0.5, embedding_dim=300, epoch=40, hidden_dim=300, lr=0.001, mode='demo', optimizer='Adam', pretrain_embedding='random', shuffle=True, test_data='data_path', train_data='data_path', update_embedding=True)
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()  # 构建系统的graph
    saver = tf.train.Saver()
    result_json = {}
    with tf.Session(config=config) as sess:
        # print('============= demo =============')
        saver.restore(sess, ckpt_file)
        if True:
            demo_sent = text
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
            else:
                demo_sent = list(demo_sent.strip())
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                # 测试代码
                print('执行model.demo，demo_data=', type(demo_data), demo_data)
                print('+===正在识别中===+')
                tag = model.demo_one(sess, demo_data)
                # 测试代码
                print('执行model.demo_one，tag=', type(tag), tag)
                PER, LOC, ORG, EQU = get_entity(tag, demo_sent)

                print('PER: {}\nLOC: {}\nORG: {}\nEQU: {}'.format(PER, LOC, ORG, EQU))
                result_json['PER'] = PER
                result_json['LOC'] = LOC
                result_json['ORG'] = ORG
                result_json['EQU'] = EQU
    return jsonify({"content": text, "result": result_json})


if __name__ == "__main__":
    app.run(debug=True, port=5050, host='0.0.0.0')
    CORS(app, supports_credentials=True)

