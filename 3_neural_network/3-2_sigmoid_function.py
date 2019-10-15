#!/usr/bin/python
# -*- Coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt

def sigmoidFunction(npX):
    u"""
    シグモイド関数

    ステップ関数を滑らかにしたもの。
    S字カーブしている。

    @param  numpy配列
    @return 0超えている要素を1として出力
    """

    return 1 / (1 + np.exp(-npX))

def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoidFunction(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1) # y軸の範囲
    plt.show()

    return 0

if __name__ == "__main__":
    exit(main())
