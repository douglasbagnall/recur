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

#if CLASSIFY_BIAS
#define CLASSIFY_RNN_FLAGS (RNN_NET_FLAG_STANDARD | RNN_COND_USE_SCALE | \
  RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR)
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

#define PERIODIC_PGM_DUMP 2047
#define PERIODIC_PGM_DUMPEES "how ihw biw"
#define PGM_DUMP_FEATURES 0

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

struct ClassifyMetadata {
  const char *classes;
  float min_freq;
  float max_freq;
  float knee_freq;
  int mfccs;
  int window_size;
  const char *basename;
  int delta_features;
  float focus_freq;
  float lag;
  int intensity_feature;
};

typedef struct _ClassifyClassEvent {
  int channel;
  int class_group;
  int window_no;
  int target;
} ClassifyClassEvent;

typedef struct _ClassifyClassGroup {
  int offset;
  int n_classes;
  char *classes;
} ClassifyClassGroup;


typedef struct _ClassifyChannel
{
  RecurNN *net;
  float *features;
  float *prev_features;
  float *pcm_now;
  float *pcm_next;
  int *group_target;
  int *group_winner;
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
  ClassifyClassGroup *class_groups;
  int n_groups;
  s16 *audio_queue;
  int read_offset;
  int write_offset;
  RecurAudioBinner *mfcc_factory;
  const char *net_filename;
  const char *basename;
  int queue_size;
  int mfccs;
  float momentum_soft_start;
  ClassifyClassEvent *class_events;
  int n_class_events;
  int class_events_index;
  int window_size;
  int window_no;
  int training;
  int random_alignment;
  int momentum_style;
  float dropout;
  float *error_weight;
  GValue *pending_properties;
  TemporalPPM *error_image;
  int delta_features;
  int intensity_feature;
  float lag;
};

struct _GstClassifyClass
{
  GstAudioFilterClass parent_class;
};

GType gst_classify_get_type(void);


G_END_DECLS
#endif /* __GST_AUDIO_CLASSIFY_H__ */
