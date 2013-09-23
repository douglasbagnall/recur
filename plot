#!/usr/bin/python
#import matplotlib
#matplotlib.use("gtkagg")
import matplotlib.pyplot as plt
#import guiqwt.pyplot as plt
#import pylab as plt
import random
import sys
import os

DEFAULT_LOGFILE = "classify.log"
DEFAULT_KEYS = ('error_sum', 'error', 'hidden_sum', 'depth',
                'error_gain', 'correct'
               )

def seek_start(f, start):
    f.seek(0, os.SEEK_END)
    end = f.tell()
    pos = 0
    step = end // 2.7
    f.seek(0)
    while True:
        pos += step
        f.seek(pos)
        print "seeking %d, step %d" % (pos, step)
        f.readline()
        line = f.readline()
        while line[:10] != 'generation':
            if not line:
                generation = 1e300
                break
            line = f.readline()

        generation = float(line[10:])

        if generation >= start:
            pos -= step
            f.seek(pos)
            if step < 10000:
                f.readline()
                return
            step //= 2.7


def read_log(fn, names,
             start=0, length=2000000, step=1):
    f = open(fn)
    if start > 10000:
        seek_start(f, start)
    if length == 0:
        length = 999999999
    gen = []
    series = {'generation': gen}
    for name in names:
        series[name] = []
    current = None
    generation = 0
    for line in f:
        try:
            name, value = line.split()
        except:
            print line
            continue
        if name == "generation":
            generation = float(value)
            continue
        if generation < start or (step > 1 and generation % step):
            continue
        if name in series:
            series[name].append(float(value))
            if current != generation:
                gen.append(generation)
                current = generation
    f.close()
    return series

def graph(lists):
    times = lists.pop('generation')
    n = 100 * len(lists) + 10
    ax = None
    subplots = []
    for k, v in lists.items():
        if ax is None:
            ax2 = ax = plt.subplot(n)
        else:
            ax2 = plt.subplot(n, sharex=ax)
        ax2.plot(times, v, 'r.', linewidth=0.5, label=k)
        #plt.legend(bbox_to_anchor=(0.2, 1.1),  loc=9)
        plt.legend(loc='upper left', numpoints=1, frameon=False, markerscale=0,borderpad=0)
        n += 1
        plt.grid(True)
    plt.show()

def summarise(lists):
    for k, v in lists.items():
        s = random.sample(v, 10)
        print k, s


def plot(logfile, args):
    numbers = [int(x) for x in args if x.isdigit()]
    keys = [x for x in args if not x.isdigit()]
    if keys == []:
        keys= DEFAULT_KEYS
    #numbers are start, length, step tuple
    lists = read_log(logfile, keys, *numbers)
    empties = [k for k, v in lists.items() if len(v) == 0]
    for k in empties:
        print "ignoring unknown key %r" % k
        del lists[k]
    #if it doesn't finish cleanly, the lists could be of differing sizes.
    shortest = min(len(x) for x in lists.values())
    for x in lists.values():
        del x[shortest:]
    print [(k, len(v)) for k,v in lists.items()]

    graph(lists)

def search_for_keys(logfile):
    from collections import defaultdict
    from math import log
    keys = defaultdict(float)
    f = open(logfile)
    i = 0
    for i, line in enumerate(f):
        try:
            k, v = line.split()
            keys[k] += 1
        except ValueError:
            break
        if i >= 100000:
            break
    for k, v in sorted(keys.iteritems()):
        print "%-20s %s" % (k, '+' * int(log(v)))





def main(args):
    logfile = DEFAULT_LOGFILE
    for x in args:
        if x.endswith('.log'):
            logfile = x
            args.remove(x)
            break

    if '--keys' in sys.argv:
        search_for_keys(logfile)
    else:
        plot(logfile, args)

main(sys.argv[1:])
