#!/usr/bin/python

import os, sys
import random
import itertools

_dirname = os.path.dirname(os.path.abspath(__file__))
os.environ['GST_PLUGIN_PATH'] = _dirname

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

CLASSES = "MEN"
DEFAULT_AUDIO_DIR = "/home/douglas/maori-language-monitoring/data/split/wav-8k/train"
TEST_AUDIO_DIR = "/home/douglas/maori-language-monitoring/data/split/wav-8k/test"

def eternal_alternator(iters, max_iterations=-1):
    cycles = [itertools.cycle(x) for x in iters]
    i = 0
    while i != max_iterations:
        for c in cycles:
            yield c.next()
        i += 1

class Classifier(object):
    trainers = None
    def __init__(self, categories=None,
                 classes=CLASSES, channels=1, report=None):
        self.mainloop = GObject.MainLoop()
        self.pipeline = Gst.Pipeline()
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect('message::eos', self.on_eos)
        self.bus.connect('message::error', self.on_error)
        self.bus.connect('message::element', self.on_element)

        self.classes = classes
        self.channels = channels
        self.report = report

        def make_add_link(el, link=None, name=None):
            x = Gst.ElementFactory.make(el, name)
            self.pipeline.add(x)
            if link is not None:
                x.link(link)
            return x

        self.sink = make_add_link('fakesink', None)
        self.classifier = make_add_link('classify', self.sink)
        self.capsfilter = make_add_link('capsfilter', self.classifier)
        self.interleave = make_add_link('interleave', self.capsfilter)
        self.filesrcs = []
        for i in range(channels):
            ac = make_add_link('audioconvert', self.interleave)
            ar = make_add_link('audioresample', ac)
            wp = make_add_link('wavparse', ar)
            fs = make_add_link('filesrc', wp)
            self.filesrcs.append(fs)

        self.classifier.set_property('classes', len(self.classes))
        self.set_channels(channels)

    def set_channels(self, channels):
        self.channels = channels
        caps =  Gst.caps_from_string("audio/x-raw, "
                                     "layout=(string)interleaved, "
                                     "channel-mask=(bitmask)0x0, "
                                     "rate=8000, channels=%s"
                                     % channels)
        self.capsfilter.set_property("caps", caps)
        self.summaries = [[]] * channels

    def train(self, data, iterations=100):
        """data is a dictionary mapping class IDs to lists of filenames.
        """
        trainers = []
        for k in self.classes:
            trainers.append([(k, x) for x in data[k]])
        self.trainers = eternal_alternator(trainers, iterations)
        self.next_training_set()
        self.pipeline.set_state(Gst.State.PLAYING)
        self.mainloop.run()

    def next_training_set(self):
        self.targets = []
        for fs in self.filesrcs:
            try:
                c, fn = self.trainers.next()
            except StopIteration:
                self.stop()
                break
            self.targets.append(c)
            fs.set_property('location', fn)
            print c, fn

        target_string = '.'.join(str(self.classes.index(x)) for x in self.targets)
        self.classifier.set_property('target', target_string)

    def identify(self, *files):
        #self.next_fileset()
        for fn, fs in zip(files, self.filesrcs):
            fs.set_property('location', fn)
        self.pipeline.set_state(Gst.State.PLAYING)
        self.mainloop.run()
        return self.winner

    def remember(self, s):
        v = s.get_value
        #print self.summaries
        for i in range(self.channels):
            winner = v('channel %d winner' % i)
            scores = tuple(v('channel %d, output %d' % (i, j))
                           for j in range(len(self.classes)))
            self.summaries[i].append((winner, scores))


    def summarise(self):
        for i, s in enumerate(self.summaries):
            #print s
            winners = [0] * len(self.classes)
            score_sums = [0] * len(self.classes)
            for w, scores in s:
                winners[w] += 1
                score_sums = [x + y for x, y in zip(score_sums, scores)]
            mean_scores = [x/ len(s) for x in score_sums]
            winner = winners.index(max(winners))
            #print self.filesrcs[i].get_property('location')
            print "channel %s  winner %s " % (i, self.classes[winner])
            #print zip(self.classes, scores)
            for a, b in zip(self.classes, scores):
                print " %s:%.2f" % (a, -b)
            self.winner = winner
        self.summaries = [[]] * self.channels


    def on_eos(self, bus, msg):
        #print('on_eos()')
        self.summarise()
        self.pipeline.set_state(Gst.State.READY)
        if self.trainers is not None:
            self.next_training_set()
            self.pipeline.set_state(Gst.State.PLAYING)
        else:
            self.stop()
            return

    def on_error(self, bus, msg):
        print('on_error():', msg.parse_error())

    def on_element(self, bus, msg):
        s = msg.get_structure()
        self.remember(s)
        if self.report:
            self.report(self, s)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.mainloop.quit()




def report_stderr(c, s):
    v = s.get_value
    error = v('error')
    print error
    for i in range(c.channels):
        winner = v('channel %d winner' % i)
        scores = tuple(v('channel %d, output %d' % (i, j))
                       for j in range(len(c.classes)))
        score_format = "%.2f " * len(c.classes)
        print c.classes, winner
        wc = c.classes[winner]
        tc = c.targets[i]
        print ("channel %d winner %s target %s "
               + score_format)  % ((i, wc, tc) + scores)


def train(_dir, cycles=1000, channels=72):
    files = [x for x in os.listdir(_dir) if x.endswith('.wav')]
    random.shuffle(files)
    categories = {}
    for c in CLASSES:
        categories[c] = [os.path.join(_dir, x) for x in files if x[0] == c]

    c = Classifier(report=report_stderr, channels=channels)
    c.train(categories, cycles)

def test(_dir=TEST_AUDIO_DIR):
    files = [x for x in os.listdir(_dir) if x.endswith('.wav')]
    random.shuffle(files)
    c = Classifier(report=None, channels=1)
    score = 0
    #files = files[:10]
    for f in files:
        print f
        winner = c.identify(os.path.join(_dir, f))
        score += winner == c.classes.find(f[0])
    print "%s/%s = %0.2f" % (score, len(files), float(score) / len(files))


def main(argv):
    if argv:
        _dir = argv[0]
    else:
        _dir = DEFAULT_AUDIO_DIR

    GObject.threads_init()
    Gst.init(None)
    #train(_dir)
    test()

main(sys.argv[1:])
