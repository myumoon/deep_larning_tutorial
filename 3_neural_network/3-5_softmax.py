#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def main():
    a = np.array([1010, 1000, 990])
    badResult = np.exp(a) / np.sum(a)
    print(u"bad : " + str(badResult)) # [inf, inf, inf]

    softMaxResult = softmax(a)
    print(u"softmax data : " + str(softMaxResult))
    print(u"softmax sum  : " + str(np.sum(a)))
    print(u"- softmaxの出力は0.0~1.0の間の実数になる")
    print(u"- その総和は1になる")
    print(u"- 総和が1になる性質は確率として解釈できる")
    print(u"- この結果から確率的な答えを出すことができる")
    print(u"- ")
    print(u"- softmax関数を適用しても各要素の大小は変わらない")
    print(u"- そして一般的にクラス分類では一番大きいニューロンに相当するクラスだけを結果にする")
    print(u"- なので出力層はsoftmax関数を省略する事が多い")

    return 0

if __name__ == "__main__":
    exit(main())
