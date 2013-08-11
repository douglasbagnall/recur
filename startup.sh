#!/bin/bash

#0 get in place
sleep 5

#move the mouse pointer to the side
xdotool mousemove 2000 300

cd /home/douglas/recur
mkdir -p nets
mkdir -p images
ln -s /home/douglas/media/video test-video || \
    echo "test-video seems to exist (that is ok)."

[[ "$DISPLAY" ]] || export DISPLAY=:0

#2 go
while true; do
    for x in *.net; do
        cp "$x" nets/$(date +%Y-%m-%d-%H-%M-%S)-$x
    done
    ./gtk-recur -f 2>> exhibition.log
    ./gtk-recur -f 2>> exhibition.log
    ./gtk-recur -f 2>> exhibition.log
done
