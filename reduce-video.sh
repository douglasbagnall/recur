#!/bin/bash

ORIG=$1
DEST=$2
FPS=$3
WIDTH=$4
HEIGHT=$5

mencoder "$ORIG" -o "$DEST" -nosound -ovc lavc -lavcopts vcodec=mpeg4 \
    -vf scale=$WIDTH:$HEIGHT -fps $FPS
