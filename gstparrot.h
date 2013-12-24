#ifndef __GST_AUDIO_PARROT_H__
#define __GST_AUDIO_PARROT_H__

#include <gst/audio/gstaudiofilter.h>
#include <gst/audio/audio.h>
#include "recur-common.h"
#include "recur-nn.h"
#include "mdct.h"
#include "mfcc.h"
#include "badmaths.h"
#include "pgm_dump.h"

G_BEGIN_DECLS

#define PARROT_RNG_SEED 11
#define PARROT_BPTT_DEPTH 30
#define LEARN_RATE 0.0003
#define MOMENTUM 0.95
#define MOMENTUM_WEIGHT 0.5

#define PARROT_MAX_CHANNELS 200
#define PARROT_MIN_CHANNELS 1
#define PARROT_RATE 16000
#define PARROT_BIAS 1
#define PARROT_BATCH_SIZE 1
#define PARROT_USE_MFCCS 0

#define PARROT_N_FFT_BINS 40
#define PARROT_WINDOW_SIZE 512

#if PARROT_USE_MFCCS
#define PARROT_N_FEATURES 20
#else
#define PARROT_N_FEATURES (PARROT_WINDOW_SIZE / 2)
#endif

#define PARROT_MFCC_MIN_FREQ 20
#define PARROT_MFCC_MAX_FREQ (PARROT_RATE * 0.499)

#if PARROT_BIAS
#define PARROT_RNN_FLAGS RNN_NET_FLAG_STANDARD
#else
#define PARROT_RNN_FLAGS RNN_NET_FLAG_NO_BIAS
#endif

#define PARROT_FORMAT "S16LE"
#define RECUR_AUDIO_BITS (16)

#define PERIODIC_SAVE_NET 1
#define TRY_RELOAD 1

#define PERIODIC_PGM_DUMP 0
#define PGM_DUMP_LEARN 0
#define PGM_DUMP_OUT 1

#define PGM_DUMP_FEATURES (PGM_DUMP_LEARN | PGM_DUMP_OUT)

#define PARROT_VALUE_SIZE 2

#define GST_TYPE_PARROT (gst_parrot_get_type())
#define GST_PARROT(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_PARROT,GstParrot))
#define GST_PARROT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_PARROT,GstParrotClass))
#define GST_IS_PARROT(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_PARROT))
#define GST_IS_PARROT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_PARROT))

/*how many chunks in the incoming buffer */
#define PARROT_QUEUE_N_CHUNKS 30

typedef struct _GstParrot GstParrot;
typedef struct _GstParrotClass GstParrotClass;

typedef struct _ParrotChannel
{
  RecurNN *train_net;
  RecurNN *dream_net;
  float *pcm_now;
  float *pcm_prev;
  float *play_now;
  float *play_prev;
  float *mdct_now;
  float *mdct_prev;
  TemporalPPM *mfcc_image;
  TemporalPPM *pcm_image;
  TemporalPPM *pcm_image2;
  TemporalPPM *dct_image;
  TemporalPPM *answer_image;
} ParrotChannel;

struct _GstParrot
{
  GstAudioFilter audiofilter;
  RecurNN *net;
  RecurNN **training_nets;
  ParrotChannel *channels;
  int n_channels;
  int queue_size;
  s16 *incoming_queue;
  int incoming_start;
  int incoming_end;
  s16 *outgoing_queue;
  int outgoing_start;
  int outgoing_end;
  mdct_lookup mdct_lut;
  RecurAudioBinner *mfcc_factory;
  float *window;

  int training;
  int playing;

  char *net_filename;
  char *pending_logfile;
  int hidden_size;
  float learn_rate;
};


struct _GstParrotClass
{
  GstAudioFilterClass parent_class;
};

GType gst_parrot_get_type(void);


G_END_DECLS
#endif /* __GST_AUDIO_PARROT_H__ */
