#!/bin/bash

ORIG=$1
DEST=$2
FPS=$3
WIDTH=$4
HEIGHT=$5
DEINT=$6

if [[ -n "$DEINT" ]]; then
    mencoder "$ORIG" -o "$DEST" -nosound -ovc lavc -lavcopts vcodec=mpeg4 \
        -vf pp=fd,scale=$WIDTH:$HEIGHT -fps $FPS
else
    mencoder "$ORIG" -o "$DEST" -nosound -ovc lavc -lavcopts vcodec=mpeg4 \
        -vf scale=$WIDTH:$HEIGHT -fps $FPS
fi
