#!/usr/bin/python
#import matplotlib
#matplotlib.use("gtkagg")
import matplotlib.pyplot as plt
#import guiqwt.pyplot as plt
#import pylab as plt
import random
import sys
import os
from collections import defaultdict
from math import log


DEFAULT_LOGFILE = "classify.log"
DEFAULT_KEYS = ('error_sum', 'error', 'hidden_sum', 'depth',
                'error_gain', 'correct',
                'weight_sum'
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
                line = 'generation 1e+300'
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


def read_log(fn, names, step=1,
             start=0, length=0):
    f = open(fn)
    if start > 10000:
        seek_start(f, start)
    if length == 0:
        length = 10 ** 9
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
            print "malformed line: %r" % line
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
                if len(gen) > length:
                    break
    f.close()
    return series

def graph(lists):
    times = lists.pop('generation')
    i = 0
    ax = None
    for k, v in lists.items():
        if ax is None:
            ax2 = ax = plt.subplot(len(lists), 1, i)
        else:
            ax2 = plt.subplot(len(lists), 1, i, sharex=ax)
        ax2.plot(times, v, 'r.', linewidth=0.5, label=k)
        plt.legend(loc='upper left', numpoints=1, frameon=False, markerscale=0,borderpad=0)
        i += 1
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
    elif all(x[0] in '-+' for x in keys):
        keys2 = list(DEFAULT_KEYS)
        for x in keys:
            op = x[0]
            k = x[1:]
            if op == '+':
                keys2.append(k)
            elif k in keys2:
                keys2.remove(k)
        keys = keys2

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
    #print [(k, len(v)) for k,v in lists.items()]

    graph(lists)

def search_for_keys(logfile):
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
    del keys['generation']
    return keys


def main(args):
    logfile = DEFAULT_LOGFILE
    for x in args:
        if x.endswith('.log'):
            logfile = x
            args.remove(x)
            break

    if '--keys' in args:
        keys = search_for_keys(logfile)
        for k, v in sorted(keys.iteritems()):
            print "%-20s %s" % (k, '+' * int(log(v)))
    elif '--all' in args:
        keys = search_for_keys(logfile)
        #print keys
        plot(logfile, keys.keys())
    else:
        plot(logfile, args)

main(sys.argv[1:])
