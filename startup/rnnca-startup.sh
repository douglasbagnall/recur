#!/bin/bash

sleep 5

#move the mouse pointer to the side, just in case
xdotool mousemove 2000 300

mkdir -p nets
mkdir -p images

[[ "$DISPLAY" ]] || export DISPLAY=:0

while true; do
    for x in rnnca*.net; do
        cp "$x" nets/$(date +%Y-%m-%d-%H-%M-%S)-$x
    done
    ./rnnca-player -f 2>> rnnca-exhibition.log
done
