#!/bin/bash

ORIG=$1
DEST=$2
FPS=$3
WIDTH=$4
HEIGHT=$5
DEINT=$6

if [[ -n "$DEINT" ]]; then
    VF="-vf mcdeint"
fi
# -an for silent
ffmpeg -r 20  -i "$ORIG" $VF -vcodec mpeg4 -s "${WIDTH}x${HEIGHT}" $DEST
