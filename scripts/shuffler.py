#!/usr/bin/python

import random
import os, sys
import re

MIN_PARA_SIZE = 300

def shuffle(text, min_para_size):
    paragraphs = re.split("\r?\n\s*\n\s*", text)
    snippets = []
    p2 = None
    for p in paragraphs:
        if p2 is not None:
            p2 += "\n" + p
            if len(p2) > min_para_size:
                snippets.append(p2)
                p2 = None
        elif len(p) > min_para_size:
            snippets.append(p)
        else:
            p2 = p

    random.shuffle(snippets)
    return '\n\n'.join(snippets)


def main(infile, outfile, min_para_size=MIN_PARA_SIZE):
    f = open(infile)
    s = shuffle(f.read(), int(min_para_size))
    f.close()
    f = open(outfile, 'w')
    f.write(s)
    f.close()

main(*sys.argv[1:])
