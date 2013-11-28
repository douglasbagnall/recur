#!/bin/bash

URI=file:///home/douglas/recur/test-video/lagos-288-192-20.avi
RECORD_URI=file:///home/douglas/recur/test-video/lagos-288-192-20-short.avi
#URI=file:///home/douglas/recur/test-video/rochester-288-192.avi

MOMENTUM=0.95
MSS=4000
LR=1e-6
HS=59
DROPOUT=0
BASENAME=$(basename $URI)
PATTERN=Y00011102C000111

for i in {1..20}; do
    if [[ $i < 3 ]]; then
        LR=1e-5
    elif [[ $i < 6 ]]; then
        LR=3e-6
    elif [[ $i < 10 ]]; then
        LR=1e-6
    else
        LR=3e-7
    fi
    for j in {1..5}; do
        echo training $i.$j LR $LR
        time gst-launch-1.0 --gst-plugin-path=. \
	    uridecodebin name=src uri=$URI \
            ! videoscale method=nearest-neighbour ! videoconvert \
            ! video/x-raw, format=I420, width=288, height=192, framerate=20/1 \
	    ! rnnca momentum-soft-start=$MSS momentum=$MOMENTUM learn-rate=$LR \
            hidden-size=$HS log-file=rnnca.log training=1 playing=0 offsets=$PATTERN \
            dropout=$DROPOUT \
	    ! fakesink
    done

    echo video $i
    time gst-launch-1.0 --gst-plugin-path=. \
        avimux name=mux ! \
        filesink location=examples/rnnca-$BASENAME-$PATTERN-$HS-$i.avi \
	uridecodebin name=src uri=$RECORD_URI \
        ! videoscale method=nearest-neighbour ! videoconvert \
        ! video/x-raw, format=I420, width=288, height=192, framerate=20/1 \
	! rnnca momentum-soft-start=$MSS momentum=$MOMENTUM learn-rate=$LR \
        hidden-size=$HS log-file=rnnca.log training=1 playing=1 offsets=$PATTERN \
        dropout=$DROPOUT \
	! videoconvert !  x264enc bitrate=512 ! mux.
done
