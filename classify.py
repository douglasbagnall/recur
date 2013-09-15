#!/usr/bin/python

import os, sys
import random

_dirname = os.path.dirname(os.path.abspath(__file__))
os.environ['GST_PLUGIN_PATH'] = _dirname
print _dirname

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
#from gi.repository import Gtk


#Gst.registry_get_default().scan_path(_dirname)

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

        def make_add(el, name=None):
            x = Gst.ElementFactory.make(el, name)
            self.pipeline.add(x)
            return x


        self.interleave = make_add('interleave')
        self.filesrcs = []
        self.wavparses = []
        for i in range(channels):
            fs = make_add('filesrc')
            wp = make_add('wavparse')
            fs.link(wp)
            wp.link(self.interleave)
            self.filesrcs.append(fs)
            self.wavparses.append(wp)

        self.classifier = make_add('classify')
        self.classifier.set_property('classes', len(self.classes))
        self.sink = make_add('fakesink')

        self.interleave.link(self.classifier)
        self.classifier.link(self.sink)
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

        #self.pipeline.seek_simple(
        #    Gst.Format.TIME,
        #    Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
        #    0
        #)

    def on_error(self, bus, msg):
        print('on_error():', msg.parse_error())

    def on_element(self, bus, msg):
        s = msg.get_structure()
        #print dir(s)
        #print s.name
        v = s.get_value
        error = v('error')
        print error
        for i in range(self.channels):
            winner = s.get_value('channel %d winner' % i)
            scores = tuple(s.get_value('channel %d, output %d' % (i, j))
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
