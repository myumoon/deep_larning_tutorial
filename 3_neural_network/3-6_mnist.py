#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os
import pprint
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
# pprint.pprint(sys.path)
from oreilliy_sample.dataset.mnist import load_mnist
from PIL import Image

def imgShow(img):
    pilImg = Image.fromarray(np.uint8(img))
    pilImg.show()

def main():
    print(u"start...")

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    print(u"x_train.shape = " + str(x_train.shape))
    print(u"t_train.shape = " + str(t_train.shape))
    print(u"x_test.shape  = " + str(x_test.shape))
    print(u"t_test.shape  = " + str(t_test.shape))

    img   = x_train[0]
    label = t_train[0]
    print(label)
    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    imgShow(img)

    return 0

if __name__ == "__main__":
    exit(main())
