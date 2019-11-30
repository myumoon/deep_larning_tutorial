#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def testNumPy():
    x = np.array([[1, 2], [3, 4]])
    y = np.array([10, 20])
    print(x * y)

def testMatplotLib():
    x = np.arange(0, 6, 0.1) # 0.1刻み
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x, y1, label = "sin")
    plt.plot(x, y2, label = "cos", linestyle = "--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("sin & cos")
    plt.legend() # 凡例
    plt.show()

def main():
    testNumPy()
    testMatplotLib()

    return 0

if __name__ == "__main__":
    exit(main())
