#ifndef __GST_AUDIO_CLASSIFY_H__
#define __GST_AUDIO_CLASSIFY_H__

#include <gst/audio/gstaudiofilter.h>
#include <gst/audio/audio.h>
#include "recur-common.h"
#include "recur-nn.h"
#include "mfcc.h"
#include "badmaths.h"
#include "pgm_dump.h"

G_BEGIN_DECLS

#define CLASSIFY_MAX_CHANNELS 1000
#define CLASSIFY_MIN_CHANNELS 1
#define CLASSIFY_RATE 8000
#define CLASSIFY_BIAS 1
#define CLASSIFY_VALUE_SIZE 2

#define CLASSIFY_N_FFT_BINS 32

#define CLASSIFY_MFCC_MIN_FREQ 200
#define CLASSIFY_MFCC_MAX_FREQ (CLASSIFY_RATE * 0.499)

#if CLASSIFY_BIAS
#define CLASSIFY_RNN_FLAGS (RNN_NET_FLAG_STANDARD | RNN_COND_USE_SCALE | \
  RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR | RNN_NET_FLAG_OWN_ACCUMULATORS)
#else
#define CLASSIFY_RNN_FLAGS (RNN_NET_FLAG_NO_BIAS | \
  RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR)
#endif

#define CLASSIFY_FORMAT "S16LE"
/*sizeof(S16LE)*/
typedef s16 audio_sample;
#define RECUR_AUDIO_BITS (8 * sizeof(audio_sample))

#define PERIODIC_SAVE_NET 0
#define TRY_RELOAD 1

#define PERIODIC_PGM_DUMP 255
#define PGM_DUMP_FEATURES 1

#define CLASSIFY_QUEUE_FACTOR 30

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

typedef struct _ClassifyClassEvent {
  int channel;
  int class;
  int window_no;
} ClassifyClassEvent;


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
  RecurNN **subnets;
  ClassifyChannel *channels;
  int n_channels;
  int n_classes;
  s16 *incoming_queue;
  int incoming_start;
  int incoming_end;
  RecurAudioBinner *mfcc_factory;
  char *net_filename;
  char *basename;
  int queue_size;
  int mfccs;
  float learn_rate;
  float momentum_soft_start;
  ClassifyClassEvent *class_events;
  int n_class_events;
  int class_events_index;
  int log_class_numbers;
  int window_size;
  int window_no;
  int mode;
  int momentum_style;
  float dropout;
  float *error_weight;
  GValue *pending_properties;
  TemporalPPM *error_image;
};


struct _GstClassifyClass
{
  GstAudioFilterClass parent_class;
};

GType gst_classify_get_type(void);


G_END_DECLS
#endif /* __GST_AUDIO_CLASSIFY_H__ */
