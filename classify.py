import os
import random
import itertools
import time

_dirname = os.path.dirname(os.path.abspath(__file__))
os.environ['GST_PLUGIN_PATH'] = _dirname

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

TMP_CLASSES = "MEN"  #Maori, English, Noise
KIWI_CLASSES = "MFN" #Male, Female, None
TMP_HIDDEN_SIZE = 499
KIWI_HIDDEN_SIZE = 199
TMP_MFCCS = 16
KIWI_MFCCS = 0

DEFAULT_LOG_FILE = "classify.log"

COLOURS = {
    "Z": "\033[00m",
    "g": '\033[00;32m',
    "G": '\033[01;32m',
    "r": '\033[00;31m',
    "R": '\033[01;31m',
    "M": "\033[01;35m",
    "P": "\033[00;35m",
    "C": "\033[01;36m",
    "Y": "\033[01;33m",
    "W": "\033[01;37m",
}

TEST_INTERVAL = 2

TRAINING = 0
TESTING = 1

SAVE_LOCATION = 'nets/autosave'

def gst_init():
    GObject.threads_init()
    Gst.init(None)

class BaseClassifier(object):
    pipeline = None
    basename = 'classify'
    def init_pipeline(self):
        self.pipeline = Gst.Pipeline()
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect('message::eos', self.on_eos)
        self.bus.connect('message::error', self.on_error)
        self.bus.connect('message::element', self.on_element)

    def make_add_link(self, el, link=None, name=None):
        if self.pipeline is None:
            self.init_pipeline()
        x = Gst.ElementFactory.make(el, name)
        self.pipeline.add(x)
        if link is not None:
            x.link(link)
        return x

    def set_channels(self, channels):
        self.channels = channels
        caps =  Gst.caps_from_string("audio/x-raw, "
                                     "layout=(string)interleaved, "
                                     "channel-mask=(bitmask)0x0, "
                                     "rate=8000, channels=%s"
                                     % channels)
        self.capsfilter.set_property("caps", caps)
        self.summaries = [[]] * channels

    def build_pipeline(self, mfccs, hsize, classes, channels, sinkname='fakesink', mic=False):
        self.classes = classes
        self.channels = channels
        self.sink = self.make_add_link(sinkname, None)
        self.classifier = self.make_add_link('classify', self.sink)
        self.classifier.set_property('mfccs', mfccs)
        self.classifier.set_property('hidden-size', hsize)
        self.classifier.set_property('classes', len(self.classes))

        self.capsfilter = self.make_add_link('capsfilter', self.classifier)
        self.interleave = self.make_add_link('interleave', self.capsfilter)
        self.filesrcs = []
        self.mic = mic
        for i in range(channels):
            ac = self.make_add_link('audioconvert', self.interleave)
            ar = self.make_add_link('audioresample', ac)
            if mic:
                fs = self.make_add_link('pulsesrc', ar)
                fs.set_property('device', 'alsa_input.pci-0000_00_1b.0.analog-stereo')

            else:
                wp = self.make_add_link('wavparse', ar)
                fs = self.make_add_link('filesrc', wp)
            self.filesrcs.append(fs)

        self.set_channels(channels)

    def __init__(self, mfccs, hsize, classes, channels=1, mainloop=None,
                 sinkname='fakesink', basename='classify', mic=False):
        if mainloop is None:
            mainloop = GObject.MainLoop()
        self.mainloop = mainloop
        self.build_pipeline(mfccs, hsize, classes, channels, sinkname=sinkname, mic=mic)
        self.basename = basename

    def on_eos(self, bus, msg):
        print('on_eos()')

    def on_error(self, bus, msg):
        print('on_error():', msg.parse_error())

    def on_element(self, bus, msg):
        s = msg.get_structure()
        if s.get_name() != "classify":
            return
        print "on_element", s.to_string()

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.mainloop.quit()


class Classifier(BaseClassifier):
    pending_files = []
    quiet = False
    def classify_files(self, *files):
        self.all_results = []
        self.pending_files = list(reversed(files))
        self.load_next_file()
        self.pipeline.set_state(Gst.State.PLAYING)
        self.mainloop.run()
        return self.all_results

    def load_next_file(self):
        fn = self.pending_files.pop()
        self.classifier.set_property("forget", 0)
        self.filesrcs[0].set_property('location', fn)
        self.file_results = []

    def on_element(self, bus, msg):
        s = msg.get_structure()
        if s.get_name() != "classify":
            return
        v = s.get_value
        winner = v('channel 0 winner')
        scores = tuple(v('channel 0, output %d' % (j))
                       for j in range(len(self.classes)))
        self.file_results.append((winner, scores))

    def collate_results(self):
        votes = [0] * len(self.classes)
        score_sums = [0] * len(self.classes)
        for w, scores in self.file_results:
            votes[w] += 1
            #scores are negative.
            score_sums = [x - y for x, y in zip(score_sums, scores)]

        mean_scores = [x / len(self.file_results) for x in score_sums]

        topscorer = mean_scores.index(max(mean_scores))
        winchar = self.classes[topscorer]
        fn = self.filesrcs[0].get_property('location')

        r = (winchar,
             mean_scores,
             votes,
             fn)
        self.all_results.append(r)
        return r

    def report(self, winchar, mean_scores, votes, fn):
        if self.quiet:
            return
        target = os.path.basename(fn)[0]
        if target == winchar:
            print (u"winner %s == %s \033[00m"
                   u"file://\033[01;32m%s\033[00m \u2714".encode('utf-8') %
                   (winchar, target, fn))
        else:
            print (u"winner %s != %s \033[00m"
                   u"file://\033[01;31m%s\033[00m \u2717".encode('utf-8') %
                   (winchar, target, fn))

        for c, v, s in zip(self.classes, votes, mean_scores):
            print (" %s: %s%4d\033[00m %s%.2f\033[00m %s%s\033[00m" %
                   (c,
                    ("\033[01;35m" if v == max(votes) else ""), v,
                    ("\033[01;36m" if s == max(mean_scores) else ""), s,
                    ("\033[01;33m" if c == target else ""),
                    '=' * int(s * 50)))

    def on_eos(self, bus, msg):
        self.pipeline.set_state(Gst.State.READY)
        r = self.collate_results()
        self.report(*r)
        if not self.pending_files:
            self.stop()
        else:
            self.load_next_file()
            self.pipeline.set_state(Gst.State.PLAYING)


def eternal_alternator(iters, max_iterations=-1):
    cycles = [itertools.cycle(x) for x in iters]
    i = 0
    while i != max_iterations:
        for c in cycles:
            yield c.next()
        i += len(cycles)

def eternal_shuffler(iters, max_iterations=-1):
    #Yield from a randomly chosen stream
    i = 0
    cycles = [itertools.cycle(x) for x in iters]
    while i != max_iterations:
        yield random.choice(cycles).next()
        i += 1

class Trainer(BaseClassifier):
    trainers = None
    lr_adjust = 1.0
    def train(self, training_data, testing_data, iterations=100, learn_rate=None,
              log_file=DEFAULT_LOG_FILE, properties=()):
        """data is a dictionary mapping class IDs to lists of filenames.
        """
        if isinstance(learn_rate, float):
            self.learn_rate = itertools.repeat(learn_rate)
        else:
            self.learn_rate = learn_rate
        self.counter = 0
        self.iterations = iterations
        trainers = []
        testers = []
        for k in self.classes:
            trainers.append([(k, x) for x in training_data[k]])
            testers.append([(k, x) for x in testing_data[k]])
        self.trainers = eternal_shuffler(trainers)
        testers = eternal_alternator(testers)
        self.testset = [testers.next() for i in range(self.channels)]
        self.test_scores = None
        self.classifier.set_property('log-file', log_file)
        for k, v in properties:
            self.classifier.set_property(k, v)
        self.timestamp = time.time()
        self.next_training_set()
        self.pipeline.set_state(Gst.State.PLAYING)
        self.mainloop.run()

    def test_set(self):
        self.test_targets = []
        self.test_scores = [0] * self.channels
        self.test_n = 0
        for x, fs in zip(self.testset, self.filesrcs):
            c, fn = x
            self.test_targets.append(self.classes.index(c))
            fs.set_property('location', fn)
            if not self.quiet:
                print c, fn
        self.classifier.set_property('forget', 0)
        self.classifier.set_property('target', '')
        self.mode = TESTING

    def evaluate_test(self):
        #print self.test_scores
        scores = [x / float(self.test_n) for x in self.test_scores]
        colours = [COLOURS[x] for x in 'PPrRYYGGgWW']
        output = []
        rightness = 0
        wrongness2 = 0
        winners = 0
        good_enough = 10 / len(self.classes)
        results = zip(self.test_targets, scores)
        results.sort()
        for t, x in results:
            i = int(x * 9.99)
            output.append(colours[i])
            output.append(self.classes[t])
            rightness += x
            winners += i > good_enough
            wrongness2 += (1.0 - x) * (1.0 - x)
        rightness /= len(scores)
        wrongness2 = (wrongness2 ** 0.5) / len(scores)
        winners /= float(len(scores))
        output += (" %s%.2f%s %.2f %s%.2f%s" %
                   (colours[int(rightness * 9.99)], rightness,
                    colours[int(10 - wrongness2 * 9.99)], wrongness2,
                    colours[int(winners * 9.99)], winners,
                    COLOURS['Z']))
        print ''.join(output)
        if winners > 0.9:
            self.save_net(tag='goodness-%d-%d' %
                          (int(rightness * 100), int(winners * 100)))
        if rightness > 0.75:
            self.lr_adjust = 0.1 / (rightness - 0.65)
        else:
            self.lr_adjust = 1.0


    def save_net(self, tag='', dir=SAVE_LOCATION):
        fn = ("%s/%s-%s-%s.net" %
              (self.basename, dir, time.time(), tag))
        print "saving %s" % fn
        self.classifier.set_property('save-net', fn)

    def next_training_set(self):
        self.targets = []
        starttime = self.timestamp
        self.timestamp = time.time()
        if self.learn_rate is not None:
            r = self.learn_rate.next() * self.lr_adjust
            print ("%s/%s learn_rate %s elapsed %.2f" %
                   (self.counter, self.iterations, r, self.timestamp - starttime))
            self.classifier.set_property('learn_rate', r)

        for fs in self.filesrcs:
            c, fn = self.trainers.next()
            self.targets.append(c)
            fs.set_property('location', fn)
            if not self.quiet:
                print c, fn

        target_string = '.'.join(str(self.classes.index(x)) for x in self.targets)
        #self.classifier.set_property('forget', 0)
        self.classifier.set_property('target', target_string)
        self.mode = TRAINING

    def on_eos(self, bus, msg):
        self.pipeline.set_state(Gst.State.READY)
        #print self.counter
        if self.test_scores:
            self.evaluate_test()
            self.test_scores = None
            self.next_training_set()
        else:
            self.classifier.set_property('save-net', '')
            self.counter += 1
            if self.counter == self.iterations:
                self.stop()
            elif self.counter % TEST_INTERVAL:
                self.next_training_set()
            else:
                self.test_set()
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_error(self, bus, msg):
        print('Error:', msg.parse_error())

    def on_element(self, bus, msg):
        total = 0
        if self.mode == TESTING:
            self.test_n += 1
            s = msg.get_structure()
            if s.get_name() != "classify":
                return
            v = s.get_value
            #print self.test_targets
            for i, t in enumerate(self.test_targets):
                winner = v('channel %d winner' % i)
                ok = winner == t
                self.test_scores[i] += ok
                total += ok
            if not self.quiet:
                score = total / float(self.channels)
                colour = COLOURS['PRRRRYYGGC'[int(score * 10)]]
                print "%s%s%s %s/%s" % (colour, "=" * total, COLOURS['Z'],
                                        total, self.channels)

        elif self.quiet:
            return
        else:
            s = msg.get_structure()
            if s.get_name() != "classify":
                return
            v = s.get_value
            error = v('error')
            print error
            fmt = "channel %d %s %s %s " + ("%.2f " * len(self.classes))
            for i in range(self.channels):
                winner = v('channel %d winner' % i)
                scores = tuple(v('channel %d, output %d' % (i, j))
                               for j in range(len(self.classes)))
                wc = self.classes[winner]
                tc = self.targets[i]
                ok = ("\033[01;32m==\033[00m" if wc == tc else "\033[01;31m!=\033[00m")
                print (fmt % ((i, wc, ok, tc) + scores))


def lr_steps(*args):
    args = list(args)
    while len(args) > 2:
        rate = args.pop(0)
        n = args.pop(0)
        for i in xrange(n):
            yield rate
    #odd number of args --> repeat forever
    while args:
        yield args[0]

def categorised_files(_dir, classes):
    files = [x for x in os.listdir(_dir) if x.endswith('.wav')]
    random.shuffle(files)
    return {c: [os.path.join(_dir, x) for x in files if x[0] == c]
            for c in classes}


class GTKClassifier(BaseClassifier):
    widget = None

    def run(self, *files):
        self.pending_files = list(reversed(files))
        self.load_next_file()
        self.pipeline.set_state(Gst.State.PLAYING)
        #self.mainloop.run()

    def load_next_file(self):
        self.pipeline.set_state(Gst.State.READY)
        if not self.mic:
            fn = self.pending_files.pop()
            self.classifier.set_property("forget", True)
            self.filesrcs[0].set_property('location', fn)
            print fn
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_element(self, bus, msg):
        if self.widget:
            s = msg.get_structure()
            if s.get_name() != "classify":
                return
            v = s.get_value
            winner = v('channel 0 winner')
            scores = tuple(-v('channel 0, output %d' % (j))
                           for j in range(len(self.classes)))
            self.widget.notify_results((winner, scores))

    def on_eos(self, bus, msg):
        if not self.pending_files:
            self.stop()
        else:
            self.load_next_file()
