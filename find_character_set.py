#/usr/bin/python
from collections import Counter
import sys

def main(files):
    chars = Counter()
    for fn in files:
        f = open(fn)
        s = ' '.join(f.read().split()).lower()
        f.close()
        chars.update(s)
    i = 1
    for c, n in chars.most_common():
        print "%2d '%s' %20d" % (i, c, n)
        i += 1

main(sys.argv[1:])
