#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def test(funcName, x1, x2, calcResult):
    result     = eval(funcName)(x1, x2)
    resultText = "success" if calcResult == result else "failure"
    print(u"{0} : {1}({2}, {3}) => {4}".format(resultText, funcName, x1, x2, result))

def calcResult(x, w, bias):
    if np.sum(x * w) + bias <= 0:
        return 0
    else:
        return 1

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    return calcResult(x, w, b)

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    return calcResult(x, w, b)

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.1
    return calcResult(x, w, b)

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y  = AND(s1, s2)
    return y

def main():
    test("AND", 0, 0, 0)
    test("AND", 0, 1, 0)
    test("AND", 1, 0, 0)
    test("AND", 1, 1, 1)
    print("-----")
    test("NAND", 0, 0, 1)
    test("NAND", 0, 1, 1)
    test("NAND", 1, 0, 1)
    test("NAND", 1, 1, 0)
    print("-----")
    test("OR", 0, 0, 0)
    test("OR", 0, 1, 1)
    test("OR", 1, 0, 1)
    test("OR", 1, 1, 1)
    print("-----")
    test("XOR", 0, 0, 0)
    test("XOR", 0, 1, 1)
    test("XOR", 1, 0, 1)
    test("XOR", 1, 1, 0)

    return 0

if __name__ == "__main__":
    exit(main())
