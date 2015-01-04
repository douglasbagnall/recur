Recur: a multimedia RNN miscellany.
-----------------------------------

Recur is a collection of Gstreamer plugins based on recurrent neural
networks, along with a character level language modeller. It began as
the technical core of an artwork, and two of the plugins (`recur` and
`rnnca`) are aimed at the rather useless task of learning to produce
abstract video. The most interesting plugin for you is probably
`classify`, which classifies audio streams. It has been used with some
success for identifying birds and human languages.

## Technical overview

The recurrent neural network (RNN) core uses rectified linear units
(ReLU) or rectified square root units. It learns via backpropagation
through time (BPTT), often using synchronic mini-batches: the weight
updates are combined from tens or hundreds of parallel streams.

The calculations are done in 32 bit floats on the CPU, and are quite
fast: on x86-64 it is significantly faster than libatlas and openblas.
Recur can achieve this by exploiting knowledge about the ReLU
architecture -- in particular, by not bothering to calculate matrix
rows that are destined to be multiplied by zero.

The data is laid out in memory to facilitate the use of SIMD
instructions, but the code avoids assembly blocks and intrinsics. This
generally works, and recent versions of GCC and Clang are able to find
reasonable SIMD solutions with minimal encouragement.

Recur was originally an artwork that learnt continuously in an effort
to recreate a video. It was working in an isolated environment (no
keyboard or network, in a distant city) for three months. It was
designed to have an interesting and uninterrupted learning *journey*,
rather than reaching a stable *end point*. Thus it has various
optional regularisers that make no sense for a destination-oriented
learner (and maybe they didn't work too well for the exhibit either).

The plugins and RNN core are written in the gnu-11 variant of C, while
many scripts are written in Python. The nets are saved using the
[CDB](http://cr.yp.to/cdb.html) format.

## Prerequisites, configuration, and compilation

The core library needs [libcdb](http://www.corpit.ru/mjt/tinycdb.html),
which will be packaged as `libcdb-dev` on Debian or `tinycdb-devel` on
Fedora.

The [Gstreamer](http://gstreamer.freedesktop.org/) plugins require
Gstreamer 1.x, Gstreamer 1.x base plugins, and Glib 2.x development
files. These are packaged with most Linux distributions, with names
like `libgstreamer1.0-dev` and `libglib2.0-dev`.

## Gstreamer plugins

There are three working plugins so far:

### recur

This one is supposed to try to learn to recreate the typical motion
and colour of the video it is watching. In fact it makes a great deal
of effort to keep changing and avoid crashing, which is somewhat at
cross-purposes to the learning.

    make libgstrecur.so
    gst-inspect-1.0  --gst-plugin-path=. recur

There is a GTK app:

    make gtk-recur
    ./gtk-recur --help

And also various example pipelines in the Makefile.

### rnnca

RNNCA stands for Recurrent Neural Network Cellular Automata. It learns
rules for a two dimensional cellular automata in imitation of the
video it is watching, and uses these to create new video.

    make libgstrnnca.so
    gst-inspect-1.0  --gst-plugin-path=. rnnca

There is a GTK app to run it:

    make rnnca-player
    ./rnnca-player --help

There is an example on [youtube](http://youtu.be/cs0w8XrpqIs).

### classify

This learns to assign class probabilities to a stream of audio. To
train it you need a lot of labelled data.

    make libgstclassify.so
    gst-inspect-1.0  --gst-plugin-path=. classify
    classify-train  --help
    classify-test  --help
    classify-gtk  --help

Documentation is slight. Sorry.

## Character level language modelling

The `text-predict` program learns to predict the next character of a
sequence of text. There are a lot of options. The defaults options
will train quickly to a cross entropy around 2.

    make text-predict
    ./text-predict --help

## Copyright and license

Copyright (C) 2014 Douglas Bagnall douglas@halo.gen.nz

This software can be distributed under the terms of the GNU Lesser
General Public License, versions 2.1 or greater, or the GNU Library
General Public License, version 2.

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
and have with various licenses, mostly very liberal. The files
`text-predict.c`, `xml-lang-classify`, `text-confabulate`,
`text-cross-entropy`, `text-classify`, `text-classify-results`, and
the contents of `ccan/opt`, are covered by the GPLv2. This does not
affect your use of the Gstreamer plugins. See licences/README for more
detail.
