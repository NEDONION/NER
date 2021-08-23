# encoding=utf-8
import os
PKL_PATH = 'word2id.pkl'

string = ''

if os.path.isfile(PKL_PATH):

    # pkl文件内容被一次性读入data
    import pickle
    reader = open(PKL_PATH, 'rb')
    data = pickle.load(reader)
    reader.close()
    print(data)

    # 将data中的数据写入字符串string
#    for lst in data:
#        x, y, w, h = list(map(int, lst))
#        x1, y1, x2, y2 = x, y, x+w, y+h
#        line = (" ").join(list(map(str, [x1, y1, x2, y2])))
#        string += (line + "\n")
#
#return string
