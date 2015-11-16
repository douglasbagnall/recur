# Copyright 2013 Douglas Bagnall <douglas@halo.gen.nz> LGPL
# -*- coding: utf-8 -*-
import os, sys
import random
import argparse
import itertools
import time
import json
import re
import numpy as np
from math import sqrt
from classify_stats import draw_roc_curve, calc_stats, draw_presence_roc
from classify_stats import actually_show_roc, calc_auc
import colour


def DEBUG(*args):
    for a in args:
        print >> sys.stderr, a
    sys.stderr.flush()

def DEBUG_LINENO(msg=''):
    import traceback
    filename, lineno, function, text = traceback.extract_stack(None, 2)[0]
    DEBUG("%s%s:%s%s %s%s()%s '%s'" % (colour.CYAN, filename, colour.BLUE,
                                       lineno, colour.CYAN, function,
                                       colour.C_NORMAL, msg))


_dirname = os.path.dirname(os.path.abspath(__file__))
os.environ['GST_PLUGIN_PATH'] = os.path.join(_dirname, 'plugins')
os.environ['GST_DEBUG_DUMP_DOT_DIR'] = '/tmp'

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

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

    def build_pipeline(self, channels, sinkname, samplerate, srcname,
                       parse_element='wavparse'):
        self.channels = channels
        self.srcname = srcname
        self.sink = self.make_add_link(sinkname, None)
        self.classifier = self.make_add_link('classify', self.sink)
        self.capsfilter = self.make_add_link('capsfilter', self.classifier)
        self.interleave = self.make_add_link('interleave', self.capsfilter)
        self.sources = []
        for i in range(channels):
            ac = self.make_add_link('audioconvert', self.interleave)
            ar = self.make_add_link('audioresample', ac)
            if srcname == 'filesrc':
                wp = self.make_add_link(parse_element, ar)
                fs = self.make_add_link(srcname, wp)
            else:
                cf = self.make_add_link('capsfilter', ar)
                cf.set_property("caps", Gst.caps_from_string("audio/x-raw, "
                                                             "layout=(string)interleaved, "
                                                             "channel-mask=(bitmask)0x0, "
                                                             "rate=%d, channels=1"
                                                             % (samplerate,)))
                fs = self.make_add_link(srcname, cf)
            self.sources.append(fs)

        caps =  Gst.caps_from_string("audio/x-raw, "
                                     "layout=(string)interleaved, "
                                     "channel-mask=(bitmask)0x0, "
                                     "rate=%d, channels=%d"
                                     % (samplerate, channels))
        self.capsfilter.set_property("caps", caps)
        if 0:
            Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL,
                                      "pipeline.dot")

    def __init__(self, channels=1, mainloop=None, sinkname='fakesink',
                 samplerate=8000, srcname='filesrc', filetype='wav'):
        parse_element = {
            'aiff': 'aiffparse',
            'au': 'auparse',
            'flac': 'flacparse',
            'auto': 'decodebin',
        }.get(filetype, 'wavparse')

        if mainloop is None:
            mainloop = GObject.MainLoop()
        self.mainloop = mainloop
        self.build_pipeline(channels, sinkname, samplerate, srcname,
                            parse_element)
        self.setp = self.classifier.set_property
        self.getp = self.classifier.get_property

    def maybe_setp(self, k, v):
        if v is not None:
            self.setp(k, v)

    def setup_from_file(self, filename, properties):
        #XXX many arguments are quietly ignored.
        self.setp('net-filename', filename)
        for k in ('ignore-start', 'features-file'):
            v = properties.get(k)
            if v is not None:
                self.setp(k, v)
        self._setup_classes()

    def _setup_classes(self, class_string=None):
        #put classes through a round trip, just to be sure it works
        if class_string is not None:
            self.setp('classes', class_string)
        self.class_groups = self.getp('classes').split(',')
        self.class_group_indices = {}
        self.classes = []
        for i, g in enumerate(self.class_groups):
            for k in g:
                self.classes.append(k)
                self.class_group_indices[k] = i

    def setup(self, properties):
        self._setup_classes(properties.pop('classes', None))
        for k, v in properties.items():
            self.setp(k, v)

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
        return {x: [0] * members for x in self.classes}


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
    no_save_net = False
    test_interval = 2
    def train(self, trainers, testers, learn_rate_fn,
              iterations=100, log_file='auto', auc_targets=None):
        self.learn_rate_fn = learn_rate_fn
        self.counter = 0
        self.save_threshold_adjust = 1.0
        self.iterations = iterations
        self.trainers = eternal_shuffler(trainers)
        testers = eternal_alternator(testers)
        self.testset = [testers.next() for i in range(self.channels)]
        self.test_scores = None
        if log_file == 'auto':
            log_file = self.getp('basename') + '.log'
        elif not log_file:
            log_file = ''
        self.setp('log-file', log_file)
        if auc_targets is None:
            if len(self.classes) == 2:
                self.auc_targets = self.classes[1]
            else:
                self.auc_targets = self.classes
        else:
            self.auc_targets = auc_targets
        self.decaying_records = [0] * (6 + len(self.auc_targets)) # for auto-save
        self.setp('load-net-now', 1)

        self.next_training_set()
        #print >> sys.stderr, "setting PLAYING()"
        self.pipeline.set_state(Gst.State.PLAYING)
        self.mainloop.run()


    def next_test_set(self):
        self.test_scores = [{x: 0 for x in y}
                            for y in self.class_groups]
        self.test_runs = [{x: 0 for x in y}
                          for y in self.class_groups]
        self.auc_lists = {c: [] for c in self.auc_targets}
        self.setp('forget', 0)
        self.setp('training', False)
        self.next_set(iter(self.testset))
        self.test_n = 0

    def next_training_set(self):
        #print >> sys.stderr, "in next_training_set()"
        self.setp('training', True)
        generation = self.getp('generation')
        #print >> sys.stderr, "generation is %d" % generation

        if self.learn_rate_fn is not None:
            r = self.learn_rate_fn(generation)
            print ("%s/%s gen %d; learn_rate %.4g;" %
                   (self.counter, self.iterations, generation, r)),
            self.setp('learn_rate', r)

        self.probability_stats = []
        for group in self.class_groups:
            self.probability_stats.append({x:([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
                                           for x in group})
        self.next_set(self.trainers)


    def next_set(self, src):
        targets = []
        self.timestamp = time.time()
        for channel, fs in enumerate(self.sources):
            f = src.next()
            targets.extend(x % channel for x in f.targets)
            fs.set_property('location', f.fullname)
            if self.verbosity > 1:
                print f.basename, f.targets

        target_string = ' '.join(targets)
        self.setp('target', target_string)
        if self.verbosity > 1:
            print target_string


    def evaluate_test(self):
        """Print something indicating how the training is going."""
        colourise = colour.colouriser(colour.SCALE_30)

        white, grey = colour.C_NORMAL, colour.GREY
        aucs = []
        for (classes, score, runs, pstats) in zip(self.class_groups,
                                                  self.test_scores,
                                                  self.test_runs,
                                                  self.probability_stats):
            # classes is a string
            # score and runs are dicts indexed by chars in classes
            # pstats is a dictionary of tuples indexed by class char.
            # Each tuple has three lists of two numbers.

            output = [self.getp('basename'), ': ']
            rightness = 0
            gap_p = 0
            ratio_p = 0
            count_p = 0
            dprime = 0
            title_colour = colour.combo(15, 235)
            t_colour = colour.combo(155, 0)
            f_colour = colour.combo(205, 0)
            for c in classes:
                output.append("%s|%s|%s" % (title_colour, c, white))
                pmeans, pvars, pcounts = pstats[c]
                wrong_p, right_p = pmeans
                wrong_c, right_c = pcounts
                wrong_var = pvars[0] / (wrong_c or 1e99)
                right_var = pvars[1] / (right_c or 1e99)
                gap_p += right_p - wrong_p
                if wrong_p:
                    ratio_p += right_p / wrong_p
                    count_p += 1

                gap = right_p - wrong_p
                dp = gap / sqrt(0.5 * (right_var + wrong_var) or 1e99)
                dprime += dp


                auc_results = self.auc_lists.get(c)
                if auc_results:
                    auc = calc_auc(auc_results)
                    output.append("%s.%03d" % (colourise(abs((auc - 0.5)) * 2.0),
                                               int(auc * 1000.0 + 0.5)))
                    aucs.append(auc)
                else:
                    output.append(".")

                s = score[c]
                r = runs[c]
                if r:
                    x = float(s) / r
                    percent = int(x * 100.0 + 0.5)
                    if r >= 10000:
                        rs = "%dk" % (int(r * 1e-3 + 0.5))
                    else:
                        rs = str(r)
                    output.append(' %s%2d%%%s/%s' % (colourise(x), percent,
                                                     grey, rs))
                    rightness += x
                else:
                    output.append(' untested ')


                output.append(" %st%s%2d%s±%02d %sf%s%2d%s±%02d " %
                              (t_colour, white, int(right_p * 99.99 + 0.5),
                               grey, int(sqrt(right_var) * 99.99 + 0.5),
                               f_colour, white,
                               int(wrong_p * 99.99 + 0.5),
                               grey, int(sqrt(wrong_var) * 99.99 + 0.5)))

            if count_p:
                ratio_p /= count_p

            mean_auc = sum(aucs) / len(aucs)
            dprime /= len(classes)
            gap_p /= len(classes)
            rightness /= len(classes)
            output.append("%s Σ %s " % (title_colour, white))

            output.append("🜚%s.%03d" %
                          (colourise((mean_auc - 0.5) * 2.0),
                           int(mean_auc * 1000.0 + 0.5)))

            output.append(" %s%2d%% %s≏%s.%02d %s×%.1f" %
                          (colourise(rightness),
                           int(rightness * 1e2 + 0.5),
                           white,
                           colourise(gap_p * 1.5),
                           int(gap_p * 1e2 + 0.5),
                           colourise(ratio_p * 0.06),
                           ratio_p))

            output.append(" %sd'%s%.2f%s" %
                          (white, colourise(dprime * 0.4), dprime, white))

            print ''.join(output)

            save = False
            for i, v in enumerate(aucs +
                                  [rightness,
                                   ratio_p,
                                   gap_p,
                                   ratio_p * gap_p,
                                   dprime,
                                   mean_auc]):
                r = self.decaying_records[i]
                if v > r:
                    save = True
                    r = v
                    if self.verbosity > 0:
                        print "%srecord %d: %.3g%s," % (grey, i, v, white),
                self.decaying_records[i] = r * 0.9995
            if save:
                self.save_named_net(tag='win-%d-gap-%d-ratio-%d-dprime-%d-auc-%d' %
                                    (int(rightness * 100 + 0.5),
                                     int(gap_p * 100 + 0.5),
                                     int(ratio_p + 0.5),
                                     int(dprime * 100 + 0.5),
                                     int(mean_auc * 1000 + 0.5)))


    def save_named_net(self, tag='', dir=SAVE_LOCATION):
        basename = self.getp('basename')
        generation = self.getp('generation')
        fn = ("%s/%s-gen-%s-%s.net" %
              (dir, basename, generation, tag))

        if not os.path.exists(dir):
            os.makedirs(dir)

        if os.path.exists(fn):
            fn = fn.replace('-gen', '-t%s-gen' % time.time())

        print "saving %s" % fn
        self.save_net(fn)

    def save_net(self, name=''):
        if not self.no_save_net:
            self.setp('save-net', name)


    def on_eos(self, bus, msg):
        self.pipeline.set_state(Gst.State.READY)
        print "elapsed %.1f" % (time.time() - self.timestamp)
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
            elif self.counter % self.test_interval:
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
            auc_lists = self.auc_lists
            self.test_n += self.channels
            v = s.get_value
            for i in range(self.channels):
                for j, group in enumerate(self.class_groups):
                    target = v('channel %d, group %d target' % (i, j))
                    if target is None:
                        continue
                    correct = v('channel %d, group %d correct' % (i, j))
                    #print group, target, correct
                    self.test_scores[j][target] += correct
                    self.test_runs[j][target] += 1
                    pstats = self.probability_stats[j]
                    for x in group:
                        correct = x == target
                        pmeans, pvars, pcounts = pstats[x]
                        n = pcounts[correct] + 1
                        pcounts[correct] = n
                        p = v('channel %d, group %d %s' % (i, j, x))
                        mean = pmeans[correct]
                        delta = p - mean
                        mean += delta / n
                        pvars[correct] += delta * (p - mean)
                        pmeans[correct] = mean
                        if x in auc_lists:
                            auc_lists[x].append((p, correct))


def lr_sqrt_exp(start, scale, min_value, post_min_value=None):
    if start > 1 or scale > 1:
        raise ValueError("learn rate start %f or scale %f is bad",
                         start, scale)

    if post_min_value is None:
        post_min_value = min_value
    if scale == 0:
        def fn(generation):
            return start
    else:
        def fn(generation):
            x = (generation * scale + 1) ** 0.5
            v = start ** x
            if v < min_value:
                return post_min_value
            return v
    return fn


def lr_inverse_time(start, min_value, offset=1.0, post_min_value=None):
    if post_min_value is None:
        post_min_value = min_value
    offset = abs(offset)
    def fn(generation):
        v = offset * start / (generation + offset)
        if v < min_value:
            return post_min_value
        return v
    return fn

def categorised_files(_dir, classes):
    files = [x for x in os.listdir(_dir) if x.endswith('.wav')]
    random.shuffle(files)
    return {c: [os.path.join(_dir, x) for x in files if x[0] == c]
            for c in classes}


class GTKClassifier(BaseClassifier):
    widget = None
    def run(self, files, reverse=False):
        self.reverse = reverse
        self.pending_files = list(reversed(files))
        self.load_next_file()
        self.pipeline.set_state(Gst.State.PLAYING)
        #self.mainloop.run()

    def load_next_file(self):
        self.pipeline.set_state(Gst.State.READY)
        if self.srcname == 'filesrc':
            fn = self.pending_files.pop()
            self.sources[0].set_property('location', fn)
            print fn
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_element(self, bus, msg):
        if self.widget:
            s = msg.get_structure()
            if s.get_name() != "classify":
                return
            v = s.get_value
            winner = v('channel 0 winner')
            scores = []
            for j, group in enumerate(self.class_groups):
                for x in group:
                    scores.extend(v('channel 0, group %d %s' % (j, x))
                                  for j in range(len(self.class_groups)))
            if self.reverse:
                scores = scores[::-1]
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


def load_binary_timings(fn, all_classes, default_state=0, classes=None,
                        threshold=0, max_duration=0):
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

    target_string = 'c%%dt%f:%s'
    group_string = '%s' + '=' * (len(all_classes) - 1)
    def add_event(state, t):
        if state is None:
            c = '-'
        else:
            c = classes[state]
        t = float(t)
        events.append((group, c, t,
                       target_string % (t, group_string % c)))

    for line in f:
        if line[0] == '[':
            calls = json.loads(line)
            name = calls.pop(0)
            events = timings.setdefault(name, [])
            state = default_state
            add_event(state, 0)
            for s, e, intensity in calls:
                if s == 0:
                    events.pop()
                if (intensity > threshold and
                    (max_duration == 0 or e - s < max_duration)):
                    add_event(1 - default_state, s)
                    add_event(default_state, e)
                else:
                    add_event(None, s)
                    add_event(default_state, e)

        else:
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


def load_multiclass_timings(fn):
    """Timings are expected to be in this form:

    <filename>','<targets>

    where <targets> is like the gstclassify.c target specification
    (try `git grep -A15 'target specification'`) but with '%d' in
    place of the channel number. That is, something like this:

    some/file.wav,c%dt0.00:A c%dt50.23:B c%d t57:-

    Too bad if your filenames contain commmas.
    """
    #XXX only works for single group
    timings = {}
    f = open(fn)
    for line in f:
        line = line.strip()
        fn, targets = line.split(',')
        events = timings.setdefault(fn, [])
        for target in targets.split():
            secs, c = target[4:].split(':')
            events.append((0, c, float(secs), target))
    f.close()
    return timings


def targeted_wav_finder(d, files):
    for fn in files:
        ffn = os.path.join(d, fn)
        if os.path.exists(ffn):
            yield (fn, ffn)

class TimedFile(object):
    def __init__(self, fn, ffn, timings=None):
        self.basename = fn.encode('utf-8')
        self.fullname = ffn.encode('utf-8')
        if timings is None:
            timings = []
        self.timings = timings
        self.targets = [x[3] for x in timings]

def always(x):
    return True

def load_timings(all_classes, timing_files, audio_directories, min_call_intensity=0,
                 max_call_duration=0, accept=always, multiclass=False):
    timings = {}
    if multiclass:
        for fn in timing_files:
            timings.update(load_multiclass_timings(fn))
    else:
        for fn in timing_files:
            classes = None
            if ',' in fn:
                fn, classes = fn.rsplit(',', 1)
                if classes not in all_classes:
                    classes = None
            timings.update(load_binary_timings(fn, all_classes, classes=classes,
                                               threshold=min_call_intensity,
                                               max_duration=max_call_duration))

    timed_files = []
    for d in audio_directories:
        for fn, ffn in targeted_wav_finder(d, timings):
            if accept(fn):
                t = TimedFile(fn, ffn, timings[fn])
                timed_files.append(t)

    return timed_files

def load_timings_from_file_names(classes, audio_directories, accept):
    group_map = {}
    for i, group in enumerate(classes):
        for x in group:
            group_map[x] = i
    timed_files = []
    for root in audio_directories:
        for d, subdirs, files in os.walk(root):
            for fn in files:
                c = fn[0]
                if accept(fn) and c in group_map:
                    ffn = os.path.join(d, fn)
                    group = group_map[c]
                    target = 'c%dt0:' + '=' * group + c + '=' * (len(classes) - group - 1)
                    timings = [(group, c, 0, target)]
                    t = TimedFile(fn, ffn, timings)
                    timed_files.append(t)

    return timed_files

def load_untimed_files(audio_directories, accept):
    untimed_files = []
    for root in audio_directories:
        for d, subdirs, files in os.walk(root):
            for fn in files:
                if accept(fn):
                    ffn = os.path.join(d, fn)
                    t = TimedFile(fn, ffn, None)
                    untimed_files.append(t)

    return untimed_files

def range_arg(bottom, top, _type=float):
    def check(x):
        x = _type(x)
        if x < bottom or x > top:
            raise argparse.ArgumentTypeError("%s is not between %s and %s" %
                                             (x, bottom, top))
        return x
    return check

def add_args_from_classifier(group, arg_names):
    classifier = Gst.ElementFactory.make('classify', 'classify-tmp')
    prop_lut = {x.name : x for x in GObject.list_properties(classifier)}
    type_lut = {
        'gchararray': str,
        'gint': int,
        'guint': int,
        'guint64': long,
        'gfloat': float,
        'gdouble': float,
        'gboolean': bool,
    }
    for args in arg_names:
        prop_name = args[-1][2:]
        prop = prop_lut[prop_name]
        prop_type = type_lut[prop.value_type.name]
        if prop_type in (float, int, long):
            prop_type = range_arg(prop.minimum, prop.maximum, prop_type)
        kwargs = {
            'type': prop_type,
            'default': None,
            'help': prop.blurb
        }
        #print args, kwargs
        if prop_type is bool and not prop.default_value:
            kwargs['action'] = 'store_true'
            del kwargs['type']

        group.add_argument(*args, **kwargs)
    return set(prop_lut)

def add_common_args(parser):
    group = parser.add_argument_group('Common arguments')

    group.add_argument('-v', '--verbosity', type=int, default=1,
                       help='0 for near silence, 2 for lots of rubbish output')
    group.add_argument('-t', '--timings', action='append',
                       help='read timings from here')
    group.add_argument('--classes-from-file-names', action='store_true',
                       help='the first letter of each file indicates its class')
    group.add_argument('-d', '--audio-directory', action='append',
                       help='find audio in this directory')
    group.add_argument('-i', '--iterations', type=int, default=10000,
                       help="how many file cycles to run for")
    group.add_argument('--min-call-intensity', type=float, default=0,
                       help="threshold for call intensity (if calls have intensity)")
    group.add_argument('--max-call-duration', type=float, default=0,
                       help="ignore calls longer than this")
    group.add_argument('--accept-file-regex', type=str, default='.+\.(wav|WAV)$',
                       help="accept files matching this regex ['.+\.(wav|WAV)$']")
    group.add_argument('--multiclass-timings', action='store_true',
                       help='the timings contain are in multiclass format')
    group.add_argument('--filetype', default='wav',
                       help='audio file type (wav, aiff, flac, auto)')

    return  add_args_from_classifier(group, (['-f', '--net-filename'],
                                             ['-c', '--classes'],
                                             ['-n', '--basename'],
                                             ['-F', '--force-load'],
                                             ['--ignore-start'],
                                             ['--features-file'],
                                             ['--features-offset'],
                                             ['--features-scale'],
                                         ))


def process_common_args(c, args, prop_names, random_seed=1, timed=True,
                        load_net=True, load_files=True):
    vargs = vars(args)
    properties = {}
    for k, v in vargs.items():
        k2 = k.replace('_', '-')
        if k2 in prop_names and v is not None:
            properties[k2] = v

    c.verbosity = args.verbosity
    if load_net:
        c.setp('force-load', args.force_load)
        if args.net_filename:
            c.setup_from_file(args.net_filename,
                              properties)
        else:
            c.setup(properties)

    if random_seed is not None:
        random.seed(random_seed)

    if not load_files:
        return None

    accept_file = re.compile(args.accept_file_regex).search

    if not timed:
        if args.audio_directory:
            files = load_untimed_files(args.audio_directory,
                                       accept_file)
        else:
            files = []
    elif args.classes_from_file_names:
        files = load_timings_from_file_names(c.class_groups,
                                             args.audio_directory,
                                             accept_file)
    else:
        files = load_timings(c.class_groups,
                             args.timings,
                             args.audio_directory,
                             args.min_call_intensity,
                             args.max_call_duration,
                             accept_file,
                             args.multiclass_timings)
    random.shuffle(files)

    return files
