#!/usr/bin/python

import sys, os
import time
import numpy as np

HERE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, HERE)
TEST_DIR = os.path.join(HERE, 'test')

import rnnumpy

def pair_123(n):
    # we go:
    # 1 0 0   1 0
    # 0 1 0   1 0
    # 0 0 1   1 0
    # 0 0 0   0 1
    #
    # After a diagonal stripe, the answer is 1.
    # There can be randomly arranged 1s in non-stripes.

    inputs = np.zeros((n, 3), dtype=np.float32)
    targets = np.zeros((n, 2), dtype=np.float32)
    choices = np.random.randint(10, size=(n))

    # blank out the first 3 rows, to prevent edge errors
    choices[:3] = 9

    inputs[choices < 3, 0] = 1.0
    inputs[1:, 1] = inputs[:-1, 0]
    inputs[2:, 2] = inputs[:-2, 0]

    for i in range(0, 2):
        inputs[choices == 3 + i, i] = 1.0


    targets[3:, 1] = inputs[:-3, 0] * inputs[1:-2, 1] * inputs[2:-1, 2]
    targets[:, 0] = 1.0 - targets[:, 1]
    return inputs, targets


def test_123():
    inputs, targets = pair_123(20000)
    test, answers = pair_123(5000)

    for h in 7, 11, 19, 39, 79:
        print '-' * 77
        print "size", h
        t1 = time.time()
        net = rnnumpy.Net(inputs.shape[1],
                          h,
                          targets.shape[1],
                          learn_rate=0.1,
                          log_file="rnnnumpy.log",
                          bptt_depth=5)
        net.train(inputs, targets, 1)

        t2 = time.time()
        print "training took %.4f" % (t2 - t1)

        results = net.classify(test)

        t1 = time.time()
        print "classifying took %.4f" % (t1 - t2)


        diff = np.abs(answers - results)

        mse = (diff ** 2).mean()

        ones_col = np.ones((diff.shape[0]), dtype=np.float32)

        stuck1 = np.zeros(diff.shape, dtype=np.float32)
        stuck2 = np.zeros(diff.shape, dtype=np.float32)
        stuck1[:, 0] = ones_col
        stuck2[:, 1] = ones_col

        wrong = np.sum(diff > 0.5) / 2

        if wrong and True:
            where_wrong = np.where(diff > 0.5)
            print where_wrong
            for y, x in zip(*where_wrong):
                if x:
                    print test[y - 4:y], results[y, x], answers[y, x]


        rand = np.random.random(diff.shape)


        print "number wrong:" , wrong

        print "squared error", mse
        print "%10s %10s %10s" % ("pattern", "vs results", "vs answers")
        print "%10s %10.2f %10.2f" % ("[1 0]",
                                  ((results - stuck1) ** 2).mean(),
                                  ((answers - stuck1) ** 2).mean())
        print "%10s %10.2f %10.2f" % ("[0 1]",
                                  ((results - stuck2) ** 2).mean(),
                                  ((answers - stuck2) ** 2).mean())
        print "%10s %10.2f %10.2f" % ("random",
                                  ((results - rand) ** 2).mean(),
                                  ((answers - rand) ** 2).mean())


def main():
    test_123()

main()
