import os, sys
import random
import itertools

_dirname = os.path.dirname(os.path.abspath(__file__))
os.environ['GST_PLUGIN_PATH'] = _dirname

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject


FILE_LENGTH=15
CLASSES = "ME"
HIDDEN_SIZE = 199

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

    def set_channels(self, channels):
        self.channels = channels
        caps =  Gst.caps_from_string("audio/x-raw, "
                                     "layout=(string)interleaved, "
                                     "channel-mask=(bitmask)0x0, "
                                     "rate=8000, channels=%s"
                                     % channels)
        self.capsfilter.set_property("caps", caps)
        self.summaries = [[]] * channels

    def build_pipeline(self, classes, channels):
        self.classes = classes
        self.channels = channels
        self.sink = self.make_add_link('fakesink', None)
        self.classifier = self.make_add_link('classify', self.sink)
        self.classifier.set_property('classes', len(self.classes))
        self.classifier.set_property('hidden-size', HIDDEN_SIZE)

        self.capsfilter = self.make_add_link('capsfilter', self.classifier)
        self.interleave = self.make_add_link('interleave', self.capsfilter)
        self.filesrcs = []
        for i in range(channels):
            ac = self.make_add_link('audioconvert', self.interleave)
            ar = self.make_add_link('audioresample', ac)
            wp = self.make_add_link('wavparse', ar)
            fs = self.make_add_link('filesrc', wp)
            self.filesrcs.append(fs)

        self.set_channels(channels)

    def __init__(self, categories=None,
                 classes=CLASSES, channels=1):
        self.mainloop = GObject.MainLoop()
        self.build_pipeline(classes, channels)

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
