Recur -- Gstreamer plugins based on recurrent neural networks
-------------------------------------------------------------

Recur began as part of an artwork, but has largely become an audio
classification system.

## Technical overview

The recurrent neural network (RNN) core uses rectified linear units
(ReLU). It learns via backpropagation through time (BPTT), often using
synchronic mini-batches: the weight updates are combined from up to
several hundred parallel streams.

The calculations are done in 32 bit floats on the CPU, and are quite
fast: at least on x86-64 it is significantly faster than libatlas and
openblas. It achieves this by exploiting knowledge about the ReLU
architecture -- in particular, by not bothering to calculate matrix
rows that are destined to be multiplied by zero.

The layout of the data is designed to facilitate the use of SIMD
instructions, while the code avoids assembly blocks and intrinsics.
Recent versions of GCC and Clang are able to find reasonable SIMD
solutions for all the important operations.

Recur was originally an artwork that learnt continuously in an effort
to recreate a video. It was working in an isolated environment (no
keyboard or network, in a distant city) for three months. It was
designed to have an interesting and uninterrupted learning *journey*,
rather than reaching a stable *end point*. Thus it has various
optional regularisers that make no sense for a destination-oriented
learner (maybe they didn't work too well for the exhibit either).

The plugins and RNN core are written in the gnu-11 variant of C, while
many scripts are written in Python. The nets are saved using the [CDB
format](http://cr.yp.to/cdb.html)

## Prerequisites, configuration, and compilation

The core library needs [libcdb](http://www.corpit.ru/mjt/tinycdb.html),
which will be packaged as `libcdb-dev` on Debian or `tinycdb-devel` on
Fedora.

The Gstreamer plugins require Gstreamer 1.x and Glib 2.x development
files. These are packaged with most Linux distributions, with names like
`libgstreamer1.0-dev` and `libglib2.0-dev`.




## Gstreamer plugins

There are three working plugins so far:

### recur

This one is supposed to try to learn to recreate the typical motion
and colour of the video it is watching. In fact it makes a great deal
of effort to keep changing and avoid crashing, which is somewhat at
cross-purposes to the learning.

    #Compile thus:
    make libgstrecur.so
    gst-inspect-1.0  --gst-plugin-path=. recur
    
    make test-pipeline

### rnnca

RNNCA stands for Recurrent Neural Network Cellular Automata. It learns
rules for a two dimensional cellular automata in imitation of the
video it is watching, and uses these to create new video.

    make libgstrnnca.so
    gst-inspect-1.0  --gst-plugin-path=. rnnca

Compile a GTK app:

    make rnnca-player
    ./rnnca-player --help

### classify

    make libgstclassify.so
    gst-inspect-1.0  --gst-plugin-path=. classify
    


## Copyright and license

Copyright (C) 2014 Douglas Bagnall douglas@halo.gen.nz

This software can be distributed under the terms of the GNU Lesser
General Public License, versions 2.1 or greater.

> This library is free software; you can redistribute it and/or
> modify it under the terms of the GNU Lesser General Public
> License as published by the Free Software Foundation; either
> version 2.1 of the License, or (at your option) any later version.
>
> This library is distributed in the hope that it will be useful,
> but WITHOUT ANY WARRANTY; without even the implied warranty of
> MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
> Lesser General Public License for more details.
>
> You should have received a copy of the GNU Lesser General Public
> License along with this library; if not, write to the Free Software
> Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

The contents of the ccan directory and mdct.c are by various authors,
and have with various licenses, mostly very liberal. The file
`test_backprop.c`, and the contents of `ccan/opt`, are covered by the
GPL. This does not affect your use of the Gstreamer plugins. See
licences/README for more detail.
