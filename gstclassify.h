#ifndef __GST_AUDIO_CLASSIFY_H__
#define __GST_AUDIO_CLASSIFY_H__

#include <gst/audio/gstaudiofilter.h>
#include <gst/audio/audio.h>
#include "recur-common.h"
#include "recur-nn.h"
#include "mfcc.h"
#include "badmaths.h"

G_BEGIN_DECLS

#define CLASSIFY_N_HIDDEN 199
#define CLASSIFY_RNG_SEED 11
#define CLASSIFY_BPTT_DEPTH 30
#define LEARN_RATE 0.0001
#define MOMENTUM 0.95
#define MOMENTUM_WEIGHT 0.5

#define CLASSIFY_MAX_CHANNELS 200
#define CLASSIFY_MIN_CHANNELS 1
#define CLASSIFY_RATE 8000
#define CLASSIFY_BIAS 1
#define CLASSIFY_BATCH_SIZE 1
#define CLASSIFY_USE_MFCCS 0
#define CLASSIFY_VALUE_SIZE 2

#define CLASSIFY_N_FFT_BINS 32

#if CLASSIFY_USE_MFCCS
#define CLASSIFY_N_FEATURES 20
#else
#define CLASSIFY_N_FEATURES CLASSIFY_N_FFT_BINS
#endif

#define CLASSIFY_MFCC_MIN_FREQ 30
#define CLASSIFY_MFCC_MAX_FREQ (CLASSIFY_RATE * 0.499)

#if CLASSIFY_BIAS
#define CLASSIFY_RNN_FLAGS RNN_NET_FLAG_STANDARD
#else
#define CLASSIFY_RNN_FLAGS RNN_NET_FLAG_NO_BIAS
#endif

#define CLASSIFY_FORMAT "S16LE"
/*sizeof(S16LE)*/
typedef s16 audio_sample;
#define RECUR_AUDIO_BITS (8 * sizeof(audio_sample))

#define PERIODIC_SAVE_NET 1
#define TRY_RELOAD 1

#define NET_LOG_FILE "classify.log"

#define PERIODIC_PGM_DUMP 511
#define REGULAR_PGM_DUMP 0
#define PGM_DUMP_FEATURES 0

#define CLASSIFY_WINDOW_SIZE 256
#define CLASSIFY_HALF_WINDOW (CLASSIFY_WINDOW_SIZE / 2)

#define CLASSIFY_QUEUE_PER_CHANNEL 20 * CLASSIFY_WINDOW_SIZE

#define GST_TYPE_CLASSIFY (gst_classify_get_type())
#define GST_CLASSIFY(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_CLASSIFY,GstClassify))
#define GST_CLASSIFY_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_CLASSIFY,GstClassifyClass))
#define GST_IS_CLASSIFY(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_CLASSIFY))
#define GST_IS_CLASSIFY_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_CLASSIFY))


typedef struct _GstClassify GstClassify;
typedef struct _GstClassifyClass GstClassifyClass;

typedef struct _ClassifyChannel
{
  RecurNN *net;
  float *features;
  float *pcm_now;
  float *pcm_next;
  int current_target;
  int current_winner;
  TemporalPPM *mfcc_image;
} ClassifyChannel;

struct _GstClassify
{
  GstAudioFilter audiofilter;
  GstAudioInfo *info;
  RecurNN *net;
  ClassifyChannel *channels;
  int n_channels;
  int n_classes;
  s16 *incoming_queue;
  int incoming_start;
  int incoming_end;
  RecurAudioBinner *mfcc_factory;
  int training;
  char *target_string;
  char *net_filename;
  int queue_size;
};


struct _GstClassifyClass
{
  GstAudioFilterClass parent_class;
};

GType gst_classify_get_type(void);


G_END_DECLS
#endif /* __GST_AUDIO_CLASSIFY_H__ */
