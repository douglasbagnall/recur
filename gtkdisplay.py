# Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL

# used by classify-gtk

import os, sys
import random

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gtk, Gdk

class ClassifierDisplay(Gtk.DrawingArea):
    results = []
    def __init__(self, classifier, size):
        Gtk.DrawingArea.__init__(self)
        self.size = size
        self.set_size_request (size, size)
        self.classifier = classifier
        self.connect('draw', self.on_draw)

    def notify_results(self, results):
        self.results.append(results)
        self.queue_draw()

    def on_draw(self, widget, cr):
        if not self.results:
            return
        cr.save()
        w = widget.get_allocated_width()
        h = widget.get_allocated_height()
        cr.translate (w / 2, h / 2)
        cr.scale(w * 0.48, h  * 0.48)

        #draw a centre line
        cr.set_line_width(0.05)
        cr.set_dash((0.01, 0.02))
        cr.set_source_rgb(0.8, 0.8, 0.8)
        cr.move_to(0, -0.5)
        cr.line_to(0, 0.4)
        cr.stroke()

        cr.set_dash(())
        step = 1.0 / len(self.results)
        i = step
        for winner, scores in self.results:
            cr.set_line_width(i * 0.08)
            i += step
            if len(scores) == 2:
                m, e = scores
                p = m - e
                cr.set_source_rgb(abs(p), m, e)
                cr.move_to(p, 0.1)
                cr.line_to(p, 0)

            elif len(scores) == 3:
                m, e, n = scores
                p = m - e
                cr.set_source_rgb(n, m, e)
                cr.move_to(p, n - 0.1)
                cr.line_to(p, n - 0.2)

            cr.stroke()
        cr.restore()
        self.results = []


def window_stop(window):
    window.classifier.stop()
    Gtk.main_quit()

def on_key_press_event(widget, event):
    keyname = Gdk.keyval_name(event.keyval).lower()
    ctrl = event.state & Gdk.ModifierType.CONTROL_MASK
    if keyname == 'q' or (ctrl and keyname == 'w'):
        window_stop(widget)
    if keyname == 'n':
        widget.classifier.load_next_file()
    if keyname == 'right':
        widget.classifier.seek_relative(5)
    if keyname == 'left':
        widget.classifier.seek_relative(-5)

    #print "Key %s (%d) was pressed" % (keyname, event.keyval)


def run(classifier, files, reverse=False):
    window = Gtk.Window()
    window.set_title ("RNN Classifier")
    app = ClassifierDisplay(classifier, 300)
    classifier.widget = app
    window.add(app)
    #XXX mucky
    window.classifier = classifier
    #print dir(window)
    window.connect_after('destroy', window_stop)
    window.connect('key_press_event', on_key_press_event)
    window.show_all()

    classifier.run(files, reverse)
    Gtk.main()
