#!/usr/bin/python

import random
import os, sys
import re

def shuffle(text):
    paragraphs = re.split("\n\s*\n\s*", text)
    random.shuffle(paragraphs)
    return '\n\n'.join(paragraphs)


def main(infile, outfile):
    f = open(infile)
    s = shuffle(f.read())
    f.close()
    f = open(outfile, 'w')
    f.write(s)
    f.close()

main(*sys.argv[1:])
