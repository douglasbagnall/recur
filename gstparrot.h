#ifndef __GST_AUDIO_PARROT_H__
#define __GST_AUDIO_PARROT_H__

#include <gst/audio/gstaudiofilter.h>
#include <gst/audio/audio.h>
#include "recur-common.h"
#include "recur-nn.h"
#include "mdct.h"
#include "mfcc.h"
#include "window.h"
#include "badmaths.h"

G_BEGIN_DECLS

#define PARROT_DRIFT 1
#define PARROT_PASSTHROUGH 0
#define PARROT_TRAIN 1

#define PARROT_N_HIDDEN 203
#define PARROT_RNG_SEED 11
#define PARROT_BPTT_DEPTH 20
#define LEARN_RATE 0.001
#define MOMENTUM 0.95
#define MOMENTUM_WEIGHT 0.5

#define PARROT_MAX_CHANNELS 20
#define PARROT_MIN_CHANNELS 1
#define PARROT_RATE 22050
#define PARROT_BIAS 1
#define PARROT_BATCH_SIZE 1
#define PARROT_USE_MFCCS 0

#define PARROT_N_FFT_BINS 40

#if PARROT_USE_MFCCS
#define PARROT_N_FEATURES 20
#else
#define PARROT_N_FEATURES PARROT_N_FFT_BINS
#endif

#define PARROT_MFCC_MIN_FREQ 20
#define PARROT_MFCC_MAX_FREQ (PARROT_RATE * 0.499)

#if PARROT_BIAS
#define PARROT_RNN_FLAGS RNN_NET_FLAG_STANDARD
#else
#define PARROT_RNN_FLAGS RNN_NET_FLAG_NO_BIAS
#endif

#define PARROT_FORMAT "S16LE"
/*sizeof(S16LE)*/
typedef s16 audio_sample;
#define RECUR_AUDIO_BITS (8 * sizeof(audio_sample))

#define PERIODIC_SAVE_NET 1
#define TRY_RELOAD 1

#define NET_LOG_FILE "parrot.log"

#define PERIODIC_PGM_DUMP 255
#define REGULAR_PGM_DUMP 0

#define PARROT_MDCT_WINDOW_BITS 9
#define PARROT_MDCT_WINDOW_SIZE (1 << PARROT_MDCT_WINDOW_BITS)
#define PARROT_HALF_WINDOW (PARROT_MDCT_WINDOW_SIZE / 2)

/*queues sizes need to be an multiple of window size */
#define PARROT_INCOMING_QUEUE_SIZE (50 * PARROT_MDCT_WINDOW_SIZE)
#define PARROT_OUTGOING_QUEUE_SIZE (50 * PARROT_MDCT_WINDOW_SIZE)

#define NET_FILENAME "parrot-" QUOTE(PARROT_N_FEATURES) "x" QUOTE(PARROT_N_HIDDEN) "-" QUOTE(PARROT_BIAS) "-" QUOTE(PARROT_RATE) "-" QUOTE(PARROT_MDCT_WINDOW_BITS) ".net"

#define GST_TYPE_PARROT (gst_parrot_get_type())
#define GST_PARROT(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_PARROT,GstParrot))
#define GST_PARROT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_PARROT,GstParrotClass))
#define GST_IS_PARROT(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_PARROT))
#define GST_IS_PARROT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_PARROT))


typedef struct _GstParrot GstParrot;
typedef struct _GstParrotClass GstParrotClass;

typedef struct _ParrotChannel
{
  RecurNN *train_net;
  RecurNN *dream_net;
  float *features1;
  float *features2;
  float *pcm1;
  float *pcm2;
  float *mdct1;
  float *mdct2;
} ParrotChannel;

struct _GstParrot
{
  GstAudioFilter audiofilter;
  GstAudioInfo *info;
  RecurNN *net;
  ParrotChannel *channels;
  int n_channels;
  s16 *incoming_queue;
  int incoming_start;
  int incoming_end;
  s16 *outgoing_queue;
  int outgoing_start;
  int outgoing_end;
  mdct_lookup mdct_lut;
  RecurAudioBinner *mfcc_factory;
  float *window;
};


struct _GstParrotClass
{
  GstAudioFilterClass parent_class;
};

GType gst_parrot_get_type(void);


G_END_DECLS
#endif /* __GST_AUDIO_PARROT_H__ */
