/* GStreamer
 * Copyright (C) <2013> Douglas Bagnall <douglas@halo.gen.nz>
 *
 */
#ifndef HAVE_RECUR_CONTEXT_H
#define HAVE_RECUR_CONTEXT_H

#define PERIODIC_PGM_DUMP 0
#define TEMPORAL_PGM_DUMP 0
#define NET_LOG_FILE "video.log"
//#define NET_LOG_FILE NULL

#define RECUR_TRAIN 1

#include <gst/gst.h>
#include "recur-common.h"
#include "recur-nn.h"
#include "mfcc.h"
#include <gst/fft/gstfftf32.h>
#include <gst/video/video.h>
#include <gst/audio/audio.h>
#include "pgm_dump.h"

#define RECUR_AUDIO_CHANNELS 1
#define RECUR_AUDIO_RATE 16000
#define RECUR_AUDIO_FORMAT "S16LE"
/*sizeof(S16LE)*/
typedef s16 audio_sample;
#define RECUR_AUDIO_BITS (8 * sizeof(audio_sample))

#define RECUR_N_FFT_BINS 40
#define RECUR_MFCC_MIN_FREQ 20
#define RECUR_MFCC_MAX_FREQ (RECUR_AUDIO_RATE * 0.499)

#define RECUR_N_MFCCS 15
#define RECUR_N_VIDEO_FEATURES ((RECUR_INPUT_HEIGHT + 2) * (RECUR_INPUT_WIDTH + 2) * 3)
#define RECUR_N_HIDDEN 199
#define RECUR_BIAS 1
#define RECUR_BPTT_DEPTH 20
#define RECUR_BATCH_SIZE 1
#define RECUR_RNG_SEED -1

#define NET_FILENAME "recur-" QUOTE(RECUR_N_HIDDEN) "-" QUOTE(RECUR_BIAS) ".net"
#define PERIODIC_SAVE_NET 1
#define TRY_RELOAD 1

#define RECUR_WORKING_WIDTH 96
#define RECUR_WORKING_HEIGHT 72

#define RECUR_INPUT_WIDTH 4
#define RECUR_INPUT_HEIGHT 3
#define RECUR_RESOLUTION_GAIN 2
#define RECUR_OUTPUT_WIDTH (RECUR_INPUT_WIDTH * RECUR_RESOLUTION_GAIN)
#define RECUR_OUTPUT_HEIGHT (RECUR_INPUT_HEIGHT * RECUR_RESOLUTION_GAIN)
#define RECUR_OUTPUT_SIZE (RECUR_OUTPUT_HEIGHT * RECUR_OUTPUT_WIDTH * 3)

#define LEARN_RATE 3e-6
#define MOMENTUM 0.9
#define MOMENTUM_WEIGHT 0.5

#define RECUR_FQ_LENGTH 16

#define RECUR_FQ_ADVANCE(f) do{ (f) = RECUR_FQ_NEXT(f);} while(0)
#define RECUR_FQ_PREVIOUS(f) (((f) - 1) & (RECUR_FQ_LENGTH - 1))
#define RECUR_FQ_NEXT(f) (((f) + 1) & (RECUR_FQ_LENGTH - 1))

#define RECUR_N_TRAINERS 12

/*all constructor constants are derived from RECUR_CONSTRUCTOR_DEPTH */
#define RECUR_AREA_GAIN (RECUR_RESOLUTION_GAIN * RECUR_RESOLUTION_GAIN)
#define RECUR_CONSTRUCTOR_DEPTH 5
#define RECUR_CONSTRUCTOR_N_LEAVES (1 << (2 * RECUR_CONSTRUCTOR_DEPTH - 2))
#define RECUR_CONSTRUCTOR_DIMENSION_GAIN (1 << (RECUR_CONSTRUCTOR_DEPTH - 1))
#define RECUR_N_CONSTRUCTORS (RECUR_CONSTRUCTOR_N_LEAVES * RECUR_AREA_GAIN / \
  (RECUR_AREA_GAIN -1))

#define RECUR_CONSTRUCTOR_WIDTH (RECUR_OUTPUT_WIDTH * RECUR_CONSTRUCTOR_DIMENSION_GAIN)
#define RECUR_CONSTRUCTOR_HEIGHT (RECUR_OUTPUT_HEIGHT * RECUR_CONSTRUCTOR_DIMENSION_GAIN)

#define RECUR_CONFAB_PLANE_SIZE (RECUR_CONSTRUCTOR_N_LEAVES \
      * RECUR_OUTPUT_WIDTH * RECUR_OUTPUT_HEIGHT)

/*recursive depth vs complexity and results.
depth leaves total nets resolution
 1       1       1       8 x 6
 2       4       5      16 x 12
 3      16      21      32 x 24
 4      64      85      64 x 48
 5     256     341     128 x 96
 6    1024    1365     256 x 192
 7    4096

leaves = 1 << (2 * depth - 2)
 total is (4 ** n - 1) / 3
       = (leaf_n * 4 / 3)

 next generation approaches 3 * total.
 (binary 1/3 is 0.0101010101..)

previous size is floor(size / 4)
*/

typedef struct _RecurTrainer {
  RecurNN *net;
  int x;
  int y;
  int scale;
} RecurTrainer;

typedef struct _RecurContext RecurContext;
typedef struct _RecurFrame RecurFrame;

struct _RecurFrame {
  u8 Y[RECUR_WORKING_WIDTH * RECUR_WORKING_HEIGHT] __attribute__((aligned (16)));
  u8 Cb[RECUR_WORKING_WIDTH * RECUR_WORKING_HEIGHT] __attribute__((aligned (16)));
  u8 Cr[RECUR_WORKING_WIDTH * RECUR_WORKING_HEIGHT] __attribute__((aligned (16)));
  GstClockTime centre_time;
  int pending;
};

struct _RecurContext {
  GstVideoInfo video_info;
  GQueue audio_queue;
  RecurAudioBinner *audio_binner;
  RecurNN *net;
  RecurFrame *frame_queue;
  int current_frame;
  int video_lag;
  u8 *planes;
  RecurNN *constructors[RECUR_N_CONSTRUCTORS];
  int fq_tail;
  int fq_head;
  RecurTrainer trainers[RECUR_N_TRAINERS];
  float learn_rate;
  float seed[RECUR_N_VIDEO_FEATURES];
  float current_audio [RECUR_N_MFCCS];
  float audio_volume;
  int osdebug;
};


void recur_queue_video_buffer(RecurContext *context, GstBuffer *buffer);
void recur_fill_video_frame(RecurContext *context, GstVideoFrame *frame);
void recur_queue_audio_segment(RecurContext *context, GstBuffer *buffer);
void recur_fill_audio_segment(RecurContext *context, GstBuffer *buffer);

void recur_context_init(RecurContext *context);
void recur_context_finalize(RecurContext *context);
void recur_context_set_video_properties(RecurContext *context, GstVideoInfo *info);

float * recur_train_rnn(RecurContext *context, RecurFrame *src_frame,
    RecurFrame *target_frame);
void recur_setup_nets(RecurContext *context, const char *log_file);

void recur_train_nets(RecurContext *context, RecurFrame *src_frame,
    RecurFrame *target_frame);

void recur_confabulate(RecurContext *context, u8 *Y, u8 *Cb, u8 *Cr);

void rnn_recursive_construct(RecurContext *context, u8 *Y, u8 *Cb, u8 *Cr,
    float *seed);

void possibly_save_state(RecurContext *context);




#endif
