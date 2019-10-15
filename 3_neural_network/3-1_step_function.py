#!/usr/bin/python
# -*- Coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

def stepFunction(npX):
    u"""
    ステップ関数

    ステップ関数は閾値を超えていたら1を返し、超えていなかったら0を返す関数。
    段階関数と呼ばれることもある。

    @param  numpy配列
    @return 0超えている要素を1として出力
    """

    # npX > 0 の構文はnumpy構文で下記のような配列変換を行う
    # [-1.0, 0.5, 2.0] => [False, True, True]
    # それを dtype=np.int で [0, 1, 1] というint配列に再変換する
    return np.array(npX > 0, dtype=np.int)

def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y = stepFunction(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y軸の範囲
    plt.show()

    return 0

if __name__ == "__main__":
    exit(main())
