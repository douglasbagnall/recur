#!/usr/bin/python3
from collections import Counter
import sys

def main(files, short=False):
    chars = Counter()
    for fn in files:
        f = open(fn)
        s = ' '.join(f.read().split()).lower()
        f.close()
        chars.update(s)
    if short:
        threshold = len(s) // 10000
        print('%-30s: {%s}    {%s}' % (' '.join(files),
                                       '' .join(c for c, n in chars.most_common()
                                                if n >= threshold),
                                       '' .join(c for c, n in chars.most_common()
                                                if n < threshold)))
    else:
        i = 1
        for c, n in chars.most_common():
            print("%2d '%s' %10d %10.3g" % (i, c, n, float(n) / len(s)))
            i += 1

if '-q' in sys.argv:
    sys.argv.remove('-q')
    main(sys.argv[1:], short=True)
else:
    main(sys.argv[1:])
