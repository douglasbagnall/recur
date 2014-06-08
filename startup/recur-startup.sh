#!/bin/bash

sleep 5

#move the mouse pointer to the side
xdotool mousemove 2000 300

mkdir -p nets
mkdir -p images

[[ "$DISPLAY" ]] || export DISPLAY=:0

while true; do
    for x in *.net; do
        cp "$x" nets/$(date +%Y-%m-%d-%H-%M-%S)-$x
    done
    ./gtk-recur -f 2>> exhibition.log
    ./gtk-recur -f 2>> exhibition.log
    ./gtk-recur -f 2>> exhibition.log
done
