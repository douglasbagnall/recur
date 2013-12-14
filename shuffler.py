#!/usr/bin/python

import random
import os, sys
import re

MIN_PARA_SIZE = 80

def shuffle(text):
    paragraphs = re.split("\n\s*\n\s*", text)
    snippets = []
    p2 = None
    for p in paragraphs:
        if p2 is not None:
            p2 += p
            if len(p2) > MIN_PARA_SIZE:
                snippets.append(p2)
                p2 = None
        elif len(p) > MIN_PARA_SIZE:
            snippets.append(p)
        else:
            p2 = p

    random.shuffle(snippets)
    return '\n\n'.join(snippets)


def main(infile, outfile):
    f = open(infile)
    s = shuffle(f.read())
    f.close()
    f = open(outfile, 'w')
    f.write(s)
    f.close()

main(*sys.argv[1:])
