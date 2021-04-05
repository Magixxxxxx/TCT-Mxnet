from utils.get_data import DATATransformer
import cv2
import mxnet as mx
from mxnet.gluon import SymbolBlock
from mxnet import nd
import numpy as np

import threading
import sys

N = 0

def load_model(path, params, gpu):
    net = SymbolBlock.imports(symbol_file=path + 'binary-cls-L-large-data-symbol.json',
                              input_names=["data"],
                              param_file=path + params,
                              ctx=mx.gpu(gpu))
    return net

class myThread (threading.Thread):
    def __init__(self, threadID, gpu, params):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.gpu = gpu
        self.transformer = DATATransformer()
        self.net = load_model('./MODEL/', params, gpu)

    def run(self):
        print ("开始线程：",self.threadID)
        global N
        while(True):
            with mutex1:
                im_path = i_f.readline().strip('\n')
            if im_path == '':break

            im = cv2.imread(im_path)
            B, G, R = cv2.split(im)
            im_tp = cv2.merge([R, G, B])
            x_ctx= self.transformer(im)
            ip = x_ctx[0].expand_dims(axis=0)

            ret = self.net(ip.as_in_context(mx.gpu(self.gpu)))

            bicls_score = ret[3].asnumpy().squeeze()
            string = '{:30} {}\n'.format(im_path, bicls_score)

            with mutex2:
                o_f.write(string)
                print(str(N),string)
                N = N+1

        print ("退出线程：",self.threadID)

if __name__ == '__main__':
    input_f = sys.argv[1]
    params = sys.argv[2]
    output_f = 'result-'+input_f
    gpus = sys.argv[3]

    gpulist = gpus.split(',')
    gpuNum = len(gpulist)

    threadNum = len(gpulist) * 8
    mutex1 = threading.Lock()
    mutex2 = threading.Lock()

    i_f = open(input_f,'r')
    o_f = open(output_f,'w')
    threadList = []

    for k in range(threadNum):
        thread = myThread(k, int(gpulist[k % gpuNum]), params)
        threadList.append(thread)
    for i in range(threadNum):
        threadList[i].start()
    for j in range(threadNum):
        threadList[j].join()

    i_f.close()
    o_f.close()
    print("处理完毕")

