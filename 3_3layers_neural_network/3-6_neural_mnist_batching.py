#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os
import pprint
import pickle
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
# pprint.pprint(sys.path)
from oreilliy_sample.dataset.mnist import load_mnist
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

"""
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
"""
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def identityFunction(x):
    return x

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y  = softmax(a3)

    return y

def getData():
    """ テストデータを返す

    Returns
        (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def initNetwork():
    # 重みとバイアスのパラメーターがdictとして入っている
    with open("oreilliy_sample/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def imgShow(img):
    pilImg = Image.fromarray(np.uint8(img))
    pilImg.show()

def main():
    print(u"start...")

    x, t = getData()
    network = initNetwork()

    print("x.shape : " + str(x.shape))
    print("x[0].shape : " + str(x[0].shape))

    def measure_notBatching():
        accuracyCnt = 0

        start    = time.time()
        for i in range(len(x)):
            y = predict(network, x[i])
            p = np.argmax(y)            # 最も確率が高い要素のインデックスを取得
            if p == t[i]:
                accuracyCnt += 1
        progress = time.time() - start

        print("not batching : " + str(progress) + "s")
        print(u"Accuracy : " + str(float(accuracyCnt) / len(x)))

    def measure_batching(batchSize):
        accuracyCnt = 0

        start    = time.time()
        for i in range(0, len(x), batchSize):
            x_batch = x[i:i + batchSize]
            y_batch = predict(network, x_batch)
            p       = np.argmax(y_batch, axis=1)           # 1次元の要素ごとに最大値を取得
            accuracyCnt += np.sum(p == t[i:i + batchSize]) # Trueの個数をカウント

        progress = time.time() - start
        print("batching : " + str(progress) + "s")
        print(u"Accuracy : " + str(float(accuracyCnt) / len(x)))

    measure_notBatching()
    measure_batching(100)

    return 0

if __name__ == "__main__":
    exit(main())
