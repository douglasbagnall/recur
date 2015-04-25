#!/usr/bin/python

import sys, os

HERE = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, HERE)
TEST_DIR = os.path.join(HERE, 'test')

import charmodel
from json import loads
from functools import partial


def parse_json_utf8(s):
    d = loads(s)
    d2 = {}
    for k, v in d.items():
        k = k.encode('utf8')
        if isinstance(v, unicode):
            v = v.encode('utf8')
        d2[k] = v
    return d2


def print_net(n):
    print "Net %s" % n
    for k in dir(n):
        print "%20s: %s" % (k, getattr(n, k))

    print "Alphabet %s" % n.alphabet
    for k in dir(n.alphabet):
        print "%20s: %s" % (k, getattr(n.alphabet, k))


def main():
    fn = TEST_DIR + '/multi-text-6c34c563i73-h99-o3650.net'
    n = charmodel.Net.load(fn, parse_json_utf8)
    print_net(n)

main()
