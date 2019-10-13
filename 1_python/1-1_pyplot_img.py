#!/usr/bin/python
# -*- Coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.image import imread


def testImgView():
    img = imread("./0_python/img/cat.jpg")
    plt.imshow(img)
    plt.show()

def main():
    testImgView()

    return 0

if __name__ == "__main__":
    exit(main())
