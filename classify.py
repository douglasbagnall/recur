# Copyright 2013 Douglas Bagnall <douglas@halo.gen.nz> LGPL
import os, sys
import random
import itertools
import time
import json
import numpy as np
from math import sqrt
from classify_stats import draw_roc_curve, calc_stats, draw_presence_roc
from classify_stats import actually_show_roc

_dirname = os.path.dirname(os.path.abspath(__file__))
os.environ['GST_PLUGIN_PATH'] = _dirname

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

MIN_FREQUENCY = 40
MAX_FREQUENCY = 3900
KNEE_FREQUENCY = 700
WINDOW_SIZE = 1024
BASENAME = 'classify'
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
        #Gst.debug_bin_to_dot_file(self.pipeline, 0, "pipeline.dot")

    def __init__(self, channels=1, mainloop=None, sinkname='fakesink', samplerate=8000):
        if mainloop is None:
            mainloop = GObject.MainLoop()
        self.mainloop = mainloop
        self.build_pipeline(channels, sinkname, samplerate)
        self.setp = self.classifier.set_property
        self.getp = self.classifier.get_property

    def setup_from_file(self, filename, **kwargs):
        #XXX many arguments are quietly ignored.
        self.setp('net-filename', filename)
        for gstarg, kwarg in (('ignore-start', 'ignore_start'),
                              ):
            val = kwargs.get(kwarg)
            if val is not None:
                self.setp(gstarg, val)
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

    def setup(self, mfccs, hsize, class_string, basename='classify',
              bottom_layer=0, window_size=None, min_freq=None,
              knee_freq=None, max_freq=None, lag=0,
              ignore_start=0, delta_features=0,
              focus_freq=0, intensity_feature=0):
        self._setup_classes(class_string)
        for gstarg, pyarg in (('window-size', window_size),
                              ('mfccs', mfccs),
                              ('hidden-size', hsize),
                              ('bottom-layer', bottom_layer),
                              ('min-frequency', min_freq),
                              ('max-frequency', max_freq),
                              ('knee-frequency', knee_freq),
                              ('focus-frequency', focus_freq),
                              ('delta-features', delta_features),
                              ('intensity-feature', intensity_feature),
                              ('lag', lag),
                              ('ignore-start', ignore_start),
                              ('basename', basename)):
            if pyarg is not None:
                self.setp(gstarg, pyarg)


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


class Classifier(BaseClassifier):
    data = []
    verbosity = 1
    ground_truth_file = None
    classification_file = None
    call_json_file = None
    score_file = None
    smooth_presence = None
    presence_file = None
    minute_results = None
    def classify(self, data,
                 ground_truth_file=None,
                 classification_file=None,
                 show_roc=False,
                 call_json_file=None,
                 call_edge_threshold=0,
                 call_peak_threshold=0,
                 call_duration_threshold=0,
                 show_presence_roc=False,
                 target_index=None,
                 summarise=False,
                 presence_index=None,
                 score_file=None,
                 smooth_presence=None,
                 presence_subsample=None,
                 presence_run_length=None,
                 presence_ignore_start=None,
                 presence_file=None,
                 presence_csv=None,
                 roc_arrows=1):
        if len(self.classes) == 2 and target_index is None:
            self.target_index = self.classes[1]
        else:
            self.target_index = target_index

        if self.target_index is None:
            self.collected_classes = self.class_group_indices.items()
        else:
            self.collected_classes = [(self.target_index,
                                      self.class_group_indices[self.target_index])]

        if ground_truth_file:
            self.ground_truth_file = open(ground_truth_file, 'w')
        if classification_file:
            self.classification_file = open(classification_file, 'w')
        if call_json_file:
            self.call_json_file = open(call_json_file, 'w')
        if score_file:
            self.score_file = open(score_file, 'w')
        self.call_edge_threshold = call_edge_threshold
        self.call_peak_threshold = call_peak_threshold
        self.call_duration_threshold = call_duration_threshold
        self.show_roc = show_roc
        self.roc_arrows = roc_arrows
        self.show_presence_roc = show_presence_roc
        self.summarise = summarise
        self.presence_index = presence_index
        self.presence_subsample = presence_subsample
        self.presence_run_length = presence_run_length
        self.presence_ignore_start = presence_ignore_start
        self.smooth_presence = smooth_presence
        if presence_file:
            self.presence_file = open(presence_file, 'w')
        if presence_csv:
            self.presence_csv = open(presence_csv, 'w')
            print >> self.presence_csv, 'filename,score,truth'

        self.data = list(reversed(data))
        self.setp('training', False)
        if self.show_roc or self.summarise:
            self.scores = {x[0]:[] for x in self.collected_classes}
        if self.show_presence_roc or self.summarise:
            self.minute_results = {x[0]:[] for x in self.collected_classes}
            self.minute_gt = {x[0]:[] for x in self.collected_classes}
        self.load_next_file()
        self.mainloop.run()

    def load_next_file(self):
        self.pipeline.set_state(Gst.State.READY)
        f = self.data.pop()
        targets = ' '.join(x % 0 for x in f.targets)
        self.current_file = f
        self.filesrcs[0].set_property('location', f.fullname)
        self.setp('target', targets)
        self.file_results = [[] for x in self.class_groups]
        self.file_scores = {x[0]:[] for x in self.collected_classes}
        self.pipeline.set_state(Gst.State.PLAYING)


    def on_element(self, bus, msg):
        s = msg.get_structure()
        if s.get_name() != "classify":
            return
        #print s.to_string()
        v = s.get_value
        timestamp = v('time')
        no_targets = not self.current_file.targets
        for k, i in self.collected_classes:
            key = 'channel 0, group %d ' % i
            correct = v(key + 'correct')
            target = v(key + 'target')
            if no_targets:
                self.file_scores[k].append((v(key + k), None, timestamp))
            elif target is None:
                continue
            else:
                self.file_scores[k].append((v(key + k), k == target, timestamp))
                if self.verbosity:
                    self.file_results[i].append((target, correct))


    def report(self):
        self.pipeline.set_state(Gst.State.READY)
        out = []
        colours = [COLOURS[x] for x in 'PPrrRRYYGgCC']
        white = COLOURS['Z']
        for groupno, file_results in enumerate(self.file_results):
            classes = self.class_groups[groupno]
            step = len(file_results) / 100.0
            next_stop = 0
            #print file_results
            for i, result in enumerate(file_results):
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

                target, correct = result
                t_index = classes.index(target)
                current_correct += correct
                current_targets[t_index] += 1

            out.extend((white, str(len(file_results)), '\n'))

        if self.target_index:
            k = self.target_index
        else:
            k = self.classes[-1]

        scores = self.file_scores[k]
        r_sum = 0
        w_sum = 0
        r_sum2, w_sum2 = 0, 0
        r_count = 0
        w_count = 0
        for s, t, time in scores:
            if t:
                r_sum += s
                r_sum2 += s * s
                r_count += 1
            else:
                w_sum += s
                w_sum2 += s * s
                w_count += 1
        if r_count:
            r_mean = r_sum / r_count
            r_stddev = (r_sum2 / r_count - r_mean * r_mean) ** 0.5
        else:
            r_mean = r_stddev = float('nan')
        if w_count:
            w_mean = w_sum / w_count
            w_stddev = (w_sum2 / w_count  - w_mean * w_mean) ** 0.5
        else:
            w_mean = w_stddev = float('nan')

        if r_count:
            diff = r_mean - w_mean
            sd = r_stddev + w_stddev
            c = colours[(diff > 0) + (diff > r_stddev) + (diff > w_stddev) +
                        (diff > sd) + (diff > sd * 2) + (diff > sd * 3) +
                        (diff > w_mean) + (diff > w_mean * 2) +
                        (diff > 0.1) + (diff > 0.5)]
        else:
            c = white
        sigma = unichr(0x03c3).encode('utf-8')
        out.append("%s scores. %s%s %.2f (%s %.2f)   not-%s %.2f (%s %.2f)%s %s\n" %
                   (k, c, k, r_mean, sigma, r_stddev, k, w_mean, sigma,
                    w_stddev, white, self.current_file.basename))

        print ''.join(out)


    def calc_presence(self, scores):
        wps = self.getp('windows-per-second')
        w_size = int(wps / (self.presence_subsample or wps) + 0.5)
        #print >> sys.stderr, "windows per second: %s. w_size %d" % (wps, w_size)

        if self.presence_run_length:
            run_length = int(wps * self.presence_run_length / w_size)
            rl_window = np.zeros(run_length) + 1.0 / run_length

        if self.presence_ignore_start is None:
            ignore_start  = 10
        else:
            ignore_start = int(self.presence_ignore_start * wps + 0.5)

        if self.presence_index is None:
            if self.presence_run_length:
                indices = [-1]
            elif self.summarise: # a historical default
                indices = [-6]
            else:
                indices = [-x * (x + 1) for x in range(1, 9)]
        else:
            indices = [-self.presence_index - 1]

        if self.target_index:
            items = [(self.target_index, scores[self.target_index])]
        else:
            items = scores.items()

        for k, v in items:
            rounding = (len(v) - ignore_start) % w_size
            v2 = v[ignore_start + rounding:]
            gt = any([x[1] for x in v2])
            s = np.array([x[0] for x in v2])
            if w_size != 1:
                s = np.mean(s.reshape(-1, w_size), 1)

            if self.presence_run_length:
                s = np.convolve(s, rl_window)

            s = np.sort(s)
            if len(s) > indices[-1]:
                r = [s[x] for x in indices]
                if self.minute_results:
                    self.minute_results[k].append(r)
                    self.minute_gt[k].append(gt)
                fn = self.current_file.basename
                if self.presence_file:
                    j = json.dumps([fn] + [round(x, 7) for x in r])
                    #print >> sys.stderr, "presence", j
                    print >> self.presence_file, j
                if self.presence_csv:
                    row = "%s,%s,%s" % (fn, r[0], gt)
                    print >> self.presence_csv, row
            else:
                print >> sys.stderr, ("ignoring presence results of length %d" %
                                      len(s))
        return indices

    def on_eos(self, bus, msg):
        if self.verbosity > 0:
            self.report()
        fn = self.current_file.basename
        scores = self.file_scores

        if self.target_index and (self.classification_file
                                  or self.ground_truth_file):
            ground_truth = [fn]
            classifications = [fn]
            for s, t, timestamp in scores[self.target_index]:
                ground_truth.append('%d' % t)
                classifications.append('%.5g' % s)

            if self.ground_truth_file:
                print >>self.ground_truth_file, ','.join(ground_truth)

            if self.classification_file:
                print >>self.classification_file, ','.join(classifications)

        if self.target_index and self.call_json_file:
            edge_threshold = self.call_edge_threshold
            peak_threshold = self.call_peak_threshold
            duration_threshold = self.call_duration_threshold
            row = [fn]
            #XXX convolve?
            start = 0
            end = 0
            score = 0
            for s, t, timestamp in scores[self.target_index]:
                if score == 0.0:
                    if s > edge_threshold:
                        start = timestamp
                        score = s
                elif s < edge_threshold:
                    if score > peak_threshold and timestamp - start > duration_threshold:
                        call = [round(start, 2), round(timestamp, 2), round(score, 4)]
                        row.append(call)
                    score = 0.0
                else:
                    score = max(score, s)

            print >>self.call_json_file, json.dumps(row)

        if self.target_index and self.score_file:
            top_scores = peak_smoothed_scores(scores[self.target_index],
                                              smooth=self.smooth_presence)
            line = [fn]
            line.extend(top_scores)
            print >>self.score_file, json.dumps(line)

        if self.show_presence_roc or self.summarise or self.presence_file:
            indices = self.calc_presence(scores)

        if self.show_roc or self.summarise:
            for k in self.scores:
                self.scores[k].extend(scores[k])

        if not self.data:
            if self.summarise and self.target_index:
                stats = calc_stats(self.scores[self.target_index],
                                   self.minute_results[self.target_index],
                                   self.minute_gt[self.target_index])
                stats['filename'] = self.getp('net-filename')
                print json.dumps(stats)

            if self.show_roc:
                if self.target_index:
                    classes = [self.target_index]
                else:
                    classes = self.classes
                for k in classes:
                    label = "%s instantaneous" % k
                    draw_roc_curve(self.scores[k], label, arrows=self.roc_arrows)
                    if self.show_presence_roc:
                        results = zip(*self.minute_results[k])
                        label_i = indices[len(indices) // 2]
                        for i, row in zip(indices, results):
                            le = (0.1 if i == label_i else 0)
                            draw_presence_roc(zip(row, self.minute_gt[k]),
                                              '%s presence %s' % (k, -i - 1),
                                              label_every=le)

                actually_show_roc(title=self.getp('basename'))
            self.stop()
        else:
            self.load_next_file()


    def on_error(self, bus, msg):
        pass


def peak_smoothed_scores(scores, top_n=200, smooth=0, ignore_first=10, kaiser=7):
    if smooth:
        window = np.kaiser(smooth, kaiser)
        s = np.array([x[0] for x in scores])
        s = np.convolve(s, window)[ignore_first:]
        top_scores = np.sort(s)[-top_n:]
        top_scores = top_scores[::-1]
    else:
        s = sorted([x[0] for x in scores[ignore_first:]], reverse=True)
        top_scores = s[:top_n]
    return top_scores

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
              iterations=100, log_file='auto', properties=(), stat_target=None):
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
        for k, v in properties:
            self.setp(k, v)
        if len(self.classes) == 2 and stat_target is None:
            stat_target = self.classes[1]
        self.stat_target = stat_target
        self.setp('load-net-now', 1)
        #Now that the net should exist, try again to set properties,
        #in case any of them require it to work (e.g. momentum).
        for k, v in properties:
            self.setp(k, v)
        self.next_training_set()
        #print >> sys.stderr, "setting PLAYING()"
        self.pipeline.set_state(Gst.State.PLAYING)
        self.mainloop.run()


    def next_test_set(self):
        self.test_scores = [{x: 0 for x in y}
                            for y in self.class_groups]
        self.test_runs = [{x: 0 for x in y}
                          for y in self.class_groups]
        self.stat_target_list = []
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
        for channel, fs in enumerate(self.filesrcs):
            f = src.next()
            targets.extend(x % channel for x in f.targets)
            fs.set_property('location', f.fullname)
            if self.verbosity > 1:
                print f.basename, targets

        target_string = ' '.join(targets)
        self.setp('target', target_string)
        if self.verbosity > 1:
            print target_string


    def evaluate_test(self):
        #print self.test_scores, self.classes, self.test_runs
        colours = [COLOURS[x] for x in 'PPrRYYGGgCCZ']
        white = COLOURS['Z']
        for (classes, score, runs, pstats) in zip(self.class_groups,
                                                  self.test_scores,
                                                  self.test_runs,
                                                  self.probability_stats):
            #classes is a string
            #score and runs are dicts indexed by chars in classes
            output = [self.getp('basename'), ': ']
            rightness = 0
            p_strings = [" means:"]
            gap_p = 0
            ratio_p = 0
            count_p = 0
            dprime = 0
            for c in classes:
                pmeans, pvars, pcounts = pstats[c]
                wrong_p, right_p = pmeans
                wrong_c, right_c = pcounts
                p_strings.append(" %s %.2f/%.2f " % (c, right_p, wrong_p))
                wrong_var = pvars[0] / (wrong_c or 1e99)
                right_var = pvars[1] / (right_c or 1e99)
                gap_p += right_p - wrong_p
                if wrong_p:
                    ratio_p += right_p / wrong_p
                    count_p += 1

                gap = right_p - wrong_p
                dp = gap / sqrt(0.5 * (right_var + wrong_var) or 1e99)
                dprime += dp

                s = score[c]
                r = runs[c]
                if r:
                    x = float(s) / r
                    i = int(x * 9.99)
                    output.append(colours[i])
                    output.append(c)
                    output.append(' %.2f%s %d/%d ' % (x, white, s, r))
                    rightness += x
                else:
                    output.append('%s --- %d/0 ' % (c, s,))
            if count_p:
                ratio_p /= count_p

            stat_list = self.stat_target_list
            stat_target = self.stat_target
            if self.stat_target in classes and stat_list:
                #auc2 = _calc_stats(stat_list[:])['auc']
                # AUC avoiding in-loop maths:
                # start off with threshold 1.0, i.e. 0% positive
                stat_list.sort(reverse=True)
                auc = 0
                tp = 0
                fp = 0
                #as the threshold drops, either a true or false positive is revealed.
                # if it is true, increment the true positive column (y++)
                # if it is false, advance along the x axis
                for p, t in stat_list:
                    if t:
                        tp += 1
                    else:
                        fp += 1
                        auc += tp
                #if the last one is true, add the last TP column
                if stat_list[-1][1]:
                    auc += tp
                area = float(tp * fp)
                auc /= area
                #print auc, auc2, auc - auc2
                auc_string = " %sAUC %s%d" %  (white,
                                              colours[max(int((auc - 0.5) * 20.0), 0)],
                                              int(auc * 1e2 + 0.5))
            else:
                auc = 0
                auc_string = ''

            dprime /= len(classes)
            gap_p /= len(classes)
            rightness /= len(classes)
            output.append(" %s%d%% %s.%02d %s\xC3\x97%.1f %sd'%s%.2f%s%s" %
                          (colours[int(rightness * 9.99)], int(rightness * 1e2 + 0.5),
                           colours[min(int(gap_p * 18.), 9)], int(gap_p * 1e2 + 0.5),
                           colours[min(int(ratio_p * 2.), 9)], ratio_p, white,
                           colours[min(int(dprime * 4.), 9)], dprime,
                           auc_string, white))

            output.extend(p_strings)

            print ''.join(output)
            adj = min(1.0, self.save_threshold_adjust)
            if (rightness > 0.8 * adj or
                ratio_p > 8.0 * adj or
                gap_p > 0.5 * adj or
                ratio_p * gap_p * adj > 2.0 or
                dprime > 1.5  * adj or
                auc > 0.94 * adj):
                self.save_threshold_adjust = 1.03
                self.save_named_net(tag='win-%d-gap-%d-ratio-%d-dprime-%d-auc-%d' %
                                    (int(rightness * 100 + 0.5),
                                     int(gap_p * 100 + 0.5),
                                     int(ratio_p + 0.5),
                                     int(dprime * 100 + 0.5),
                                     int(auc * 100 + 0.5)))
            else:
                self.save_threshold_adjust *= 0.99


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
            stat_target = self.stat_target
            stat_list = self.stat_target_list
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
                        if x == stat_target:
                            stat_list.append((p, correct))

def negate_exponent(x, name=None):
    """Sometimes, on the command-line, I write 1e6 when I mean 1e-6, and
    where the former value would be ridiculous, meaningless, or
    destructive. This function corrects those errors -- silently
    unless a name is provided.
    """
    m, e = ("%.9e" % x).split('e')
    s = e[0]
    e = e[1:]
    if s == '-':
        y = float(m + 'e+' + e)
    elif s == '+':
        y = float(m + 'e-' + e)
    else:
        y = x #should not happen

    if name is not None:
        print >> sys.stderr, "assuming %s %e means %e" % (name, x, y)
    return y

def lr_sqrt_exp(start, scale, min_value, post_min_value=None):
    if start > 1:
        start = negate_exponent(start, 'learn-rate')
    if scale > 1:
        scale = negate_exponent(scale, 'learn-rate-scale')
    if post_min_value is None:
        post_min_value = min_value
    if scale == 0:
        def fn(generation):
            return start
    else:
        def fn(generation):
            x = (generation * scale + 1) ** 0.5
            v = start ** x
            #print >> sys.stderr, "start %f, x %f, v %f" % (start, x, v)
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

def load_timings(all_classes, timing_files, audio_directories, min_call_intensity=0,
                 max_call_duration=0):
    timings = {}
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
            t = TimedFile(fn, ffn, timings[fn])
            timed_files.append(t)

    return timed_files

def load_timings_from_file_names(classes, audio_directories):
    group_map = {}
    for i, group in enumerate(classes):
        for x in group:
            group_map[x] = i
    timed_files = []
    for root in audio_directories:
        for d, subdirs, files in os.walk(root):
            for fn in files:
                c = fn[0]
                if fn.endswith('.wav') and c in group_map:
                    ffn = os.path.join(d, fn)
                    group = group_map[c]
                    target = 'c%dt0:' + '=' * group + c + '=' * (len(classes) - group - 1)
                    timings = [(group, c, 0, target)]
                    t = TimedFile(fn, ffn, timings)
                    timed_files.append(t)

    return timed_files

def load_untimed_files(audio_directories):
    untimed_files = []
    for root in audio_directories:
        for d, subdirs, files in os.walk(root):
            for fn in files:
                if fn.endswith('.wav'):
                    ffn = os.path.join(d, fn)
                    t = TimedFile(fn, ffn, None)
                    untimed_files.append(t)

    return untimed_files


def add_common_args(parser):
    group = parser.add_argument_group('Common arguments')
    group.add_argument('-v', '--verbosity', type=int, default=1,
                       help='0 for near silence, 2 for lots of rubbish output')
    group.add_argument('-t', '--timings', action='append',
                       help='read timings from here')
    group.add_argument('--classes-from-file-names', action='store_true',
                       help='the first letter of each file indicates its class')
    group.add_argument('-f', '--net-filename',
                       help='load RNN from here')
    group.add_argument('-d', '--audio-directory', action='append',
                       help='find audio in this directory')
    group.add_argument('-i', '--iterations', type=int, default=10000,
                       help="how many file cycles to run for")
    group.add_argument('-H', '--hidden-size', type=int,
                       help="number of hidden neurons")
    group.add_argument('-B', '--bottom-layer', type=int,
                       help="number of bottom layer output nodes")
    group.add_argument('-c', '--classes', default='tf',
                       help="classes (letter per class, groups separated by commas)")
    group.add_argument('-w', '--window-size', default=WINDOW_SIZE, type=int,
                       help="size of the FFT window")
    group.add_argument('-n', '--basename', default=BASENAME,
                       help="save nets etc using this basename")
    group.add_argument('-F', '--force-load', action='store_true',
                       help="load the net even if metadata doesn't match")
    group.add_argument('--delta-features', type=int,
                       help="use this many layers of derivitive features")
    group.add_argument('--intensity-feature', action='store_true',
                       help="use the overall intensity as a feature")
    group.add_argument('--lag', type=float, default=0.0,
                       help="add this much lag to loaded times")
    group.add_argument('--ignore-start', type=float, default=0.0,
                       help="ignore this many seconds at start of file")
    group.add_argument('--focus-frequency', type=float, default=0.0,
                       help="focus on frequencies around this")
    group.add_argument('--min-frequency', type=float, default=MIN_FREQUENCY,
                       help="lowest audio frequency to consider")
    group.add_argument('--max-frequency', type=float, default=MAX_FREQUENCY,
                       help="highest audio frequency to consider")
    group.add_argument('--knee-frequency', type=float, default=KNEE_FREQUENCY,
                       help="higher for more top-end response")
    group.add_argument('--mfccs', type=int, default=0,
                       help="How many MFCCs to use (0 for raw fft bins)")
    group.add_argument('--min-call-intensity', type=float, default=0,
                       help="threshold for call intensity (if calls have intensity)")
    group.add_argument('--max-call-duration', type=float, default=0,
                       help="ignore calls longer than this")

def process_common_args(c, args, random_seed=1, timed=True, load=True):
    c.verbosity = args.verbosity
    if load:
        c.setp('force-load', args.force_load)
        if args.net_filename:
            c.setup_from_file(args.net_filename,
                              ignore_start=args.ignore_start)
        else:
            c.setup(args.mfccs,
                    args.hidden_size,
                    args.classes,
                    window_size=args.window_size,
                    bottom_layer=args.bottom_layer,
                    basename=args.basename,
                    min_freq=args.min_frequency,
                    max_freq=args.max_frequency,
                    knee_freq=args.knee_frequency,
                    focus_freq=args.focus_frequency,
                    lag=args.lag,
                    ignore_start=args.ignore_start,
                    delta_features=args.delta_features,
                    intensity_feature=args.intensity_feature)

    if random_seed is not None:
        random.seed(random_seed)

    if not timed:
        if args.audio_directory:
            files = load_untimed_files(args.audio_directory)
        else:
            files = []
    elif args.classes_from_file_names:
        files = load_timings_from_file_names(c.class_groups,
                                             args.audio_directory)
    else:
        files = load_timings(c.class_groups,
                             args.timings,
                             args.audio_directory,
                             args.min_call_intensity,
                             args.max_call_duration)
    random.shuffle(files)
    return files
