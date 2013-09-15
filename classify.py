#!/usr/bin/python

import os, sys
import random

_dirname = os.path.dirname(os.path.abspath(__file__))
os.environ['GST_PLUGIN_PATH'] = _dirname

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

CLASSES = "MEN"
DEFAULT_AUDIO_DIR = "/home/douglas/maori-language-monitoring/data/split/wav-8k/train"
TEST_AUDIO_DIR = "/home/douglas/maori-language-monitoring/data/split/wav-8k/test"

class Classifier(object):
    def __init__(self, _dir, ext='.wav', training=True, classes=CLASSES, channels=1):
        self.mainloop = GObject.MainLoop()
        self.pipeline = Gst.Pipeline()
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect('message::eos', self.on_eos)
        self.bus.connect('message::error', self.on_error)
        self.bus.connect('message::element', self.on_element)

        self.training = training
        self.classes = classes
        self.channels = channels

        self.dir = _dir
        self.pending_files = [x for x in os.listdir(_dir) if x.endswith(ext)]
        random.shuffle(self.pending_files)

        def make_add_link(el, link=None, name=None):
            x = Gst.ElementFactory.make(el, name)
            self.pipeline.add(x)
            if link is not None:
                x.link(link)
            return x

        self.sink = make_add_link('fakesink', None)
        self.classifier = make_add_link('classify', self.sink)
        self.capsfilter = make_add_link('capsfilter', self.classifier)
        caps =  Gst.caps_from_string("audio/x-raw, "
                                     "layout=(string)interleaved, "
                                     "rate=8000, channels=%s"
                                     % self.channels)
        self.capsfilter.set_property("caps", caps)
        self.interleave = make_add_link('interleave', self.capsfilter)
        self.filesrcs = []
        for i in range(channels):
            ac = make_add_link('audioconvert', self.interleave)
            ar = make_add_link('audioresample', ac)
            wp = make_add_link('wavparse', ar)
            fs = make_add_link('filesrc', wp)
            self.filesrcs.append(fs)

        self.classifier.set_property('classes', len(self.classes))
        self.next_fileset()


    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        self.mainloop.run()

    def next_fileset(self):
        targets = []
        self.targets = []
        for fs in self.filesrcs:
            try:
                fn = self.pending_files.pop()
            except IndexError:
                self.stop()
                return
            lang = fn[0]
            fullpath = os.path.join(self.dir, fn)
            print fullpath
            fs.set_property('location', fullpath)
            targets.append(str(self.classes.index(lang)))
            self.targets.append(lang)
        if self.training:
            self.classifier.set_property('target', '.'.join(targets))

    def on_eos(self, bus, msg):
        print('on_eos()')
        self.pipeline.set_state(Gst.State.READY)
        self.next_fileset()
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_error(self, bus, msg):
        print('on_error():', msg.parse_error())

    def on_element(self, bus, msg):
        s = msg.get_structure()
        v = s.get_value
        error = v('error')
        print error,
        for i in range(self.channels):
            winner = v('channel %d winner' % i)
            scores = tuple(v('channel %d, output %d' % (i, j))
                           for j in range(len(self.classes)))
            score_format = "%.2f " * len(self.classes)
            wc = self.classes[winner]
            tc = self.targets[i]
            print ("channel %d winner %s target %s "
                   + score_format)  % ((i, wc, tc) + scores)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)
        self.mainloop.quit()



def main(argv):
    if argv:
        dir = argv[0]
    else:
        dir = DEFAULT_AUDIO_DIR

    GObject.threads_init()
    Gst.init(None)

    c = Classifier(dir)
    c.run()

def test():
    GObject.threads_init()
    Gst.init(None)
    c = Classifier(TEST_AUDIO_DIR, training=False)
    c.run()


main(sys.argv[1:])
#test()
