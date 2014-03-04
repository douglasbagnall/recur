import os
import random
import itertools
import time

_dirname = os.path.dirname(os.path.abspath(__file__))
os.environ['GST_PLUGIN_PATH'] = _dirname

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

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

SAVE_LOCATION = 'nets/autosave'

def gst_init():
    GObject.threads_init()
    Gst.init(None)

class BaseClassifier(object):
    pipeline = None
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

    def build_pipeline(self, channels, sinkname, samplerate):
        self.channels = channels
        self.sink = self.make_add_link(sinkname, None)
        self.classifier = self.make_add_link('classify', self.sink)
        self.capsfilter = self.make_add_link('capsfilter', self.classifier)
        self.interleave = self.make_add_link('interleave', self.capsfilter)
        self.filesrcs = []
        for i in range(channels):
            ac = self.make_add_link('audioconvert', self.interleave)
            ar = self.make_add_link('audioresample', ac)
            wp = self.make_add_link('wavparse', ar)
            fs = self.make_add_link('filesrc', wp)
            self.filesrcs.append(fs)

        self.channels = channels
        caps =  Gst.caps_from_string("audio/x-raw, "
                                     "layout=(string)interleaved, "
                                     "channel-mask=(bitmask)0x0, "
                                     "rate=%d, channels=%d"
                                     % (samplerate, channels))
        self.capsfilter.set_property("caps", caps)

    def __init__(self, channels=1, mainloop=None, sinkname='fakesink', samplerate=8000):
        if mainloop is None:
            mainloop = GObject.MainLoop()
        self.mainloop = mainloop
        self.build_pipeline(channels, sinkname, samplerate)
        self.setp = self.classifier.set_property
        self.getp = self.classifier.get_property

    def setup_from_file(self, filename):
        self.setp('net-filename', filename)
        self.classes = self.getp('classes').split(',')

    def setup(self, mfccs, hsize, class_string, basename='classify',
              bottom_layer=0, window_size=None, min_freq=None,
              knee_freq=None, max_freq=None):
        #put classes through a round trip, just to be sure it works
        self.setp('classes', class_string)
        self.classes = self.getp('classes').split(',')
        if window_size is not None:
            self.setp('window-size', window_size)
        if mfccs is not None:
            self.setp('mfccs', mfccs)
        if hsize is not None:
            self.setp('hidden-size', hsize)
        if bottom_layer:
            self.setp('bottom-layer', bottom_layer)
        if min_freq is not None:
            self.setp('min-frequency', min_freq)
        if max_freq is not None:
            self.setp('max-frequency', max_freq)
        if knee_freq is not None:
            self.setp('knee-frequency', knee_freq)
        self.setp('basename', basename)

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

    def get_results_counter(self, members=2):
        groups = []
        for group in self.classes:
            groups.append({x: [0] * members for x in group})
        return groups



class Classifier(BaseClassifier):
    data = []
    quiet = False
    ground_truth_file = None
    classification_file = None
    def classify(self, data,
                 ground_truth_file=None,
                 classification_file=None, show_roc=False):
        if ground_truth_file:
            self.ground_truth_file = open(ground_truth_file, 'w')
            self.ground_truth_index = (0, self.classes[0][-1])
        if classification_file:
            self.classification_file = open(classification_file, 'w')
            self.classification_index = (0, self.classes[0][-1])

        self.show_roc = show_roc
        self.all_results = []
        self.class_results = self.get_results_counter()
        self.data = list(reversed(data))
        self.setp('training', False)
        self.load_next_file()
        self.mainloop.run()
        return self.all_results

    def load_next_file(self):
        self.pipeline.set_state(Gst.State.READY)
        fn, timings = self.data.pop()
        targets = ' '.join(x[3] % 0 for x in timings)
        #print fn, targets
        self.current_file = fn
        self.filesrcs[0].set_property('location', fn)
        self.setp('target', targets)
        self.file_results = [[] for x in self.classes]
        self.file_class_results = self.get_results_counter()
        self.probabilities = self.get_results_counter(0)
        self.ground_truth = self.get_results_counter(0)

        self.pipeline.set_state(Gst.State.PLAYING)


    def on_element(self, bus, msg):
        s = msg.get_structure()
        if s.get_name() != "classify":
            return
        #print s.to_string()
        v = s.get_value
        for i, group in enumerate(self.class_results):
            f_group = self.file_class_results[i]
            probs = self.probabilities[i]
            key = 'channel 0, group %d ' % i
            correct = v(key + 'correct')
            winner = v(key + 'winner')
            target = v(key + 'target')
            for k in group:
                probs[k].append(v(key + k))
                if self.ground_truth_file:
                    self.ground_truth[i][k].append(k == target)

            group[target][correct] += 1
            f_group[target][correct] += 1
            self.file_results[i].append((target, correct))


    def report(self):
        self.pipeline.set_state(Gst.State.READY)
        out = []
        colours = [COLOURS[x] for x in 'PPrrRRYYGgCC']

        for groupno, file_results in enumerate(self.file_results):
            classes = self.classes[groupno]
            step = len(file_results) / 100.0
            next_stop = 0
            for i, result in enumerate(file_results):
                target, correct = result
                t_index = classes.index(target)
                if i >= next_stop:
                    if i:
                        s = sum(current_targets)
                        m = max(current_targets)
                        if m > s * 0.9:
                            c = classes[current_targets.index(m)]
                        else:
                            c = '~'
                        colour = colours[int(current_correct * 10.01 / s)]
                        out.append('%s%s' % (colour, c))
                    next_stop += step
                    current_correct = 0
                    current_targets = [0] * len(classes)
                current_correct += correct
                current_targets[t_index] += 1

            out.extend((COLOURS['Z'],'\n'))

        print ''.join(out)
        #print self.file_class_results
        #self.show_roc_curve()
        self.pipeline.set_state(Gst.State.PLAYING)


    def show_roc_curve(self):
        import matplotlib.pyplot as plt
        for probabilities in self.probabilities:
            kp = sorted(probabilities).values()
            del probabilities[:]
            sum_true = sum(1 for x in kp if x[1])
            sum_false = len(kp) - sum_true

            tp_scale = 1.0 / (sum_true or 1)
            fp_scale = 1.0 / (sum_false or 1)
            tp = []
            fp = []
            false_positives = sum_false
            true_positives = sum_true
            half = 0
            ax, ay, ad, ap = 0, 0, 0, 0
            bx, by, bd, bp = 0, 0, 99, 0
            cx, cy, cd, cp = 0, 0, 0, 0
            for prob, truth in kp:
                false_positives -= not truth
                true_positives -= truth
                x = false_positives * fp_scale
                y = true_positives * tp_scale
                half += prob < 0.5
                d = (1 - x) * (1 - x) + y * y
                if d > ad:
                    ad = d
                    ax = x
                    ay = y
                    ap = prob
                d = x * x + (1 - y) * (1 - y)
                if d < bd:
                    bd = d
                    bx = x
                    by = y
                    bp = prob
                d = y - x
                if d > cd:
                    cd = d
                    cx = x
                    cy = y
                    cp = prob
                fp.append(x)
                tp.append(y)

            if half < len(fp):
                hx = (fp[half - 1] + fp[half]) * 0.5
                hy = (tp[half - 1] + tp[half]) * 0.5
            else:
                hx = fp[half - 1]
                hy = tp[half - 1]


            fp.reverse()
            tp.reverse()
            print "best", bx, by
            print "half", hx, hy
            plt.plot(fp, tp)
            plt.annotate("0.5", (hx, hy), (0.4, 0.4),
                         arrowprops={'width':1, 'color': '#00cc00'})
            plt.annotate("furthest from all bad %.2g" % ap, (ax, ay), (0.3, 0.3),
                         arrowprops={'width':1, 'color': '#00cccc'},
                         )
            plt.annotate("closest to all good %.2g" % bp, (bx, by), (0.6, 0.6),
                         arrowprops={'width':1, 'color': '#cc0000'},
                         )
            plt.annotate("furthest from diagonal %.2g" % cp, (cx, cy), (0.5, 0.5),
                         arrowprops={'width':1, 'color': '#aa6600'},
                         )

            plt.axes().set_aspect('equal')
        plt.show()

    def on_eos(self, bus, msg):
        #self.report()
        fn = os.path.basename(self.current_file)
        if self.ground_truth_file:
            i, k = self.ground_truth_index
            a = [fn] + ['%d' % x for x in self.ground_truth[i][k]]
            print >>self.ground_truth_file, ','.join(a)

        if self.classification_file:
            i, k = self.ground_truth_index
            a = [fn] + ['%.5g' % x for x in self.probabilities[i][k]]
            print >>self.classification_file, ','.join(a)


        if not self.data:
            if self.show_roc:
                self.show_roc_curve()
            self.stop()
        else:
            self.load_next_file()


    def on_error(self, bus, msg):
        pass


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
    no_save_net = False
    def train(self, trainers, testers, timings, iterations=100, learn_rate=None,
              dropout=0.0, log_file='auto', properties=()):
        if isinstance(learn_rate, (int, float)):
            self.learn_rate = itertools.repeat(learn_rate)
        else:
            self.learn_rate = learn_rate
        if isinstance(dropout, (int, float)):
            self.dropout = itertools.repeat(dropout)
        else:
            self.dropout = dropout
        self.counter = 0
        self.iterations = iterations
        self.timings = timings
        self.trainers = eternal_shuffler(trainers)
        testers = eternal_alternator(testers)
        self.testset = [testers.next() for i in range(self.channels)]
        self.test_scores = None
        if log_file == 'auto':
            log_file = self.getp('basename') + '.log'
        elif not log_file:
            log_file = ''
        self.setp('log-file', log_file)
        for k, v in properties:
            self.setp(k, v)
        self.timestamp = time.time()
        self.next_training_set()
        self.pipeline.set_state(Gst.State.PLAYING)
        self.mainloop.run()


    def next_test_set(self):
        self.test_scores = [{x: 0 for x in y}
                            for y in self.classes]
        self.test_runs = [{x: 0 for x in y}
                          for y in self.classes]
        self.setp('dropout', 0)
        self.setp('forget', 0)
        self.setp('training', False)
        self.next_set(iter(self.testset))
        self.test_n = 0
        self.timestamp = time.time()

    def next_training_set(self):
        dropout = self.dropout.next()
        self.setp('dropout', dropout)
        self.setp('training', True)
        starttime = self.timestamp
        self.timestamp = time.time()

        if self.learn_rate is not None:
            r = self.learn_rate.next() * self.lr_adjust
            print ("%s/%s learn_rate %.4g dropout %.2g elapsed %.2f" %
                   (self.counter, self.iterations, r, dropout,
                    self.timestamp - starttime))
            self.setp('learn_rate', r)


        self.probability_sums = []
        self.probability_counts = []
        for group in self.classes:
            self.probability_sums.append({x:[0.0, 0.0] for x in group})
            self.probability_counts.append({x:[0.0, 0.0] for x in group})

        self.next_set(self.trainers)


    def next_set(self, src):
        targets = []
        for channel, fs in enumerate(self.filesrcs):
            fn = src.next()
            timings = self.timings.get(fn, [])
            for group, c, t, ts in timings:
                targets.append(ts % channel)
            fs.set_property('location', fn)
            if not self.quiet:
                print fn, timings

        target_string = ' '.join(targets)
        self.setp('target', target_string)
        if not self.quiet:
            print target_string


    def evaluate_test(self):
        #print self.test_scores, self.classes, self.test_runs
        colours = [COLOURS[x] for x in 'PPrRYYGGgCC']
        for classes, score, runs, probs, pcounts in zip(self.classes,
                                                        self.test_scores,
                                                        self.test_runs,
                                                        self.probability_sums,
                                                        self.probability_counts):
            #classes is a string
            #score and runs are dicts indexed by chars in classes
            output = []
            rightness = 0
            winners = 0
            good_enough = 10 // len(classes)

            for c in classes:
                #print c, score, runs
                s = score[c]
                r = runs[c]
                if r:
                    x = float(s) / r
                    i = int(x * 9.99)
                    output.append(colours[i])
                    output.append(c)
                    output.append(' %.2f%s %d/%d ' % (x, COLOURS['Z'], s, r))
                    rightness += x
                    winners += i > good_enough
                else:
                    output.append('%s --- %d/0 ' % (c, s,))

            rightness /= len(classes)
            winners /= float(len(classes))
            output.append(" %s%.2f%s %.2f%s" %
                          (colours[int(rightness * 9.99)], rightness,
                           colours[int(winners * 9.99)], winners,
                           COLOURS['Z']))

            output.append(" probabilities (right/wrong)")
            for x in classes:
                #print x, probs, pcounts
                wrong, right = probs[x]
                wrong_c, right_c = pcounts[x]
                right_p = right / right_c if right_c else float('nan')
                wrong_p = wrong / wrong_c if wrong_c else float('nan')
                output.append(" %s %.3f/%.3f " % (x, right_p, wrong_p))

            print ''.join(output)
            if winners > 0.9:
                self.save_named_net(tag='goodness-%d-%d' %
                                    (int(rightness * 100), int(winners * 100)))


    def save_named_net(self, tag='', dir=SAVE_LOCATION):
        basename = self.classifier.getproperty('basename')
        fn = ("%s/%s-%s-%s.net" %
              (dir, basename, time.time(), tag))
        print "saving %s" % fn
        self.save_net(fn)

    def save_net(self, name=''):
        if not self.no_save_net:
            self.setp('save-net', name)


    def on_eos(self, bus, msg):
        self.pipeline.set_state(Gst.State.READY)
        #print self.counter
        if self.test_scores:
            self.evaluate_test()
            self.test_scores = None
            self.next_training_set()
        else:
            self.save_net()
            self.counter += 1
            if self.counter == self.iterations:
                self.stop()
            elif self.counter % TEST_INTERVAL:
                self.next_training_set()
            else:
                self.next_test_set()
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_error(self, bus, msg):
        print('Error:', msg.parse_error())


    def on_element(self, bus, msg):
        s = msg.get_structure()
        #print s.to_string()
        name = s.get_name()
        if name == 'classify' and not self.getp('training'):
            self.test_n += self.channels
            v = s.get_value
            for i in range(self.channels):
                for j, group in enumerate(self.classes):
                    target = v('channel %d, group %d target' % (i, j))
                    correct = v('channel %d, group %d correct' % (i, j))
                    #print group, target, correct
                    self.test_scores[j][target] += correct
                    self.test_runs[j][target] += 1
                    for x in group:
                        p = v('channel %d, group %d %s' % (i, j, x))
                        #print x, target, p
                        self.probability_sums[j][x][x == target] += p
                        self.probability_counts[j][x][x == target] += 1.0



def lr_steps(*args):
    args = list(args)
    while len(args) > 2:
        rate = float(args.pop(0))
        n = int(args.pop(0))
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
        fn = self.pending_files.pop()
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
            #correct = v('channel 0 correct')
            #target = v('channel 0 target')
            #print correct, winner, target, winner == target
            #print s.to_string()
            scores = []
            for j, group in enumerate(self.classes):
                for x in group:
                    scores.extend(-v('channel 0, group %d %s' % (j, x))
                                  for j in range(len(self.classes)))

            self.widget.notify_results((winner, scores))

    def on_eos(self, bus, msg):
        if not self.pending_files:
            self.stop()
        else:
            self.load_next_file()

    def seek_relative(self, secs):
        p = self.pipeline
        now = p.query_position(Gst.Format.TIME)[1]
        print "%.1f" % (now * 1e-9)
        then = max(0, now + secs * (10 ** 9))
        p.seek_simple(Gst.Format.TIME, 0, then)


def load_binary_timings(fn, all_classes, default_state=0, classes=None):
    #all_classes is a sequence of class groups. Each group of
    #all_classes is a string of characters. By default the first one
    #is used, but if a line like 'group: Xxy' comes along, the the
    #group represented by Xxy is used.

    #start with classes[0], first time switches to classes[1],
    #alternating thereafter
    f = open(fn)
    timings = {}
    group = 0
    if classes == None:
        classes = all_classes[0]

    print "loading %s with classes %s" % (fn, classes)
    target_string = 'c%%dt%s:%s'
    group_string = '%s' + '=' * (len(all_classes) - 1)
    def add_event(state, t):
        c = classes[state]
        events.append((group, classes[state], float(t),
                       target_string % (t, group_string % c)))

    for line in f:
        d = line.split()
        name = d.pop(0)
        if name == 'group:':
            classes = d[0]
            if classes not in all_classes:
                raise ValueError("%s refers to unknown class group '%s'", fn, classes)
            group = all_classes.index(classes)
            group_string = '=' * group + '%s' + '=' * (len(all_classes) - group - 1)
        else:
            events = timings.setdefault(name, [])

            if d:
                state = default_state
                if float(d[0]) > 0:
                    add_event(state, 0)
                for t in d:
                    state = 1 - state
                    add_event(state, t)
            else:
                add_event(default_state, 0)

    f.close()
    #XXX sort timings?
    return timings


def targeted_wav_finder(d, files):
    for fn in files:
        ffn = os.path.join(d, fn)
        if os.path.exists(ffn):
            yield (fn, ffn)



def load_timings(all_classes, timing_files, audio_directories):
    timings = {}
    for fn in timing_files:
        classes = None
        if ',' in fn:
            fn, classes = fn.rsplit(',', 1)
            if classes not in all_classes:
                classes = None
        timings.update(load_binary_timings(fn, all_classes, classes=classes))

    #timings = coalesce_timings(timings)

    timed_files = []
    for d in audio_directories:
        timed_files.extend(targeted_wav_finder(d, timings))

    random.shuffle(timed_files)

    #print timings
    full_timings = {ffn: timings[fn] for fn, ffn in timed_files}

    return timed_files, full_timings

def add_common_args(parser, WINDOW_SIZE, BASENAME):
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='lots of rubbish output')
    parser.add_argument('-t', '--timings', action='append',
                        help='read timings from here')
    parser.add_argument('-f', '--net-filename',
                        help='load RNN from here')
    parser.add_argument('-d', '--audio-directory', action='append',
                        help='find audio in this directory')
    parser.add_argument('-i', '--iterations', type=int, default=10000,
                        help="how many file cycles to run for")
    parser.add_argument('-H', '--hidden-size', type=int,
                        help="number of hidden neurons")
    parser.add_argument('-B', '--bottom-layer', type=int,
                        help="number of bottom layer output nodes")
    parser.add_argument('-c', '--classes', default='tf',
                        help="classes (letter per class, groups separated by commas)")
    parser.add_argument('-w', '--window-size', default=WINDOW_SIZE, type=int,
                        help="size of the FFT window")
    parser.add_argument('-n', '--basename', default=BASENAME,
                        help="save nets etc using this basename")
