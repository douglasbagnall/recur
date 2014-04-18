#!/usr/bin/python
#Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL/MPL2

import sys
import struct, array
import cdb

def read_int(n, key, unsigned=False):
    v = n.get(key)
    format = ' bh i   q'[len(v)]
    if format == ' ':
        raise ValueError("%r has invalid size %d" % (key, len(v)))
    if unsigned:
        format = format.upper()
    return struct.unpack('<' + format, v)[0]


def read(fn):
    n = cdb.init(fn)
    version = read_int(n, 'save_format_version')
    if version >= 4:
        useful_keys = {'net.i_size', 'net.h_size', 'net.o_size', 'net.input_size',
                       'net.hidden_size', 'net.output_size', 'net.ih_size', 'net.ho_size',
                       'net.bias', 'net.flags', 'net.ih_weights', 'net.ho_weights',
                       'bottom_layer.input_size', 'bottom_layer.output_size',
                       'bottom_layer.i_size', 'bottom_layer.o_size',
                       'bottom_layer.overlap', 'bottom_layer.weights'}
    else:
        useful_keys = {'bias', 'depth', 'flags', 'h_size',
                       'hidden_size', 'ho_size', 'ho_weights',
                       'i_size', 'ih_size', 'ih_weights',
                       'input_size', 'o_size', 'output_size'}

    print "keys are: (* means possibly useful)"
    for k in sorted(n.keys()):
        print "  %s %s" % ('*' if k in useful_keys else ' ', k)

def main():
    read(sys.argv[1])

main()
