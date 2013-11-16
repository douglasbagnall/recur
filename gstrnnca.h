#ifndef __GOT_RNNCA_H__
#define __GOT_RNNCA_H__

#include <gst/video/gstvideofilter.h>
#include "recur-common.h"
#include "recur-nn.h"
#include "badmaths.h"

G_BEGIN_DECLS

#define TRY_RELOAD 1
#define RNNCA_BIAS 1
#define RNNCA_N_TRAINERS 70
#define RNNCA_WIDTH 128
#define RNNCA_HEIGHT 96

#define RNNCA_BATCH_SIZE 1
#define RNNCA_RNG_SEED 11
#define RNNCA_BPTT_DEPTH 30
#define MOMENTUM 0.95
#define MOMENTUM_WEIGHT 0.5

#if RNNCA_BIAS
#define RNNCA_RNN_FLAGS (RNN_NET_FLAG_STANDARD)
#else
#define RNNCA_RNN_FLAGS (RNN_NET_FLAG_NO_BIAS)
#endif

#define PERIODIC_PGM_DUMP 511
#define PERIODIC_SAVE_NET 255

#define PERIODIC_CHECK_STASIS 0
#define PERIODIC_SHUFFLE_TRAINERS 3
#define PGM_DUMP_CHANGED_MASK 0

const int RNNCA_YUV_OFFSETS[] = {
  -1, -1,   0, -1,   1, -1,
  -1,  0,            1,  0,
  -1,  1,   0,  1,   1,  1
};

const int RNNCA_Y_ONLY_OFFSETS[] = {
       -1, -2,  1, -2,
  -2, -1,            2, -1,
  -2,  1,            2,  1,
       -1,  2,  1,  2
};

const int RNNCA_YUV_LEN = sizeof(RNNCA_YUV_OFFSETS) / sizeof(RNNCA_YUV_OFFSETS[0]);
const int RNNCA_Y_ONLY_LEN = sizeof(RNNCA_Y_ONLY_OFFSETS) / sizeof(RNNCA_Y_ONLY_OFFSETS[0]);

#define RNNCA_N_FEATURES (((RNNCA_YUV_LEN * 3 + RNNCA_Y_ONLY_LEN) >> 1) + 2)

typedef struct _RnncaFrame {
  u8 *Y;
  u8 *Cb;
  u8 *Cr;
} RnncaFrame;


typedef struct _RnncaTrainer {
  RecurNN *net;
  int x;
  int y;
} RnncaTrainer;


typedef struct _GstRnnca GstRnnca;
typedef struct _GstRnncaClass GstRnncaClass;

struct _GstRnnca
{
  GstVideoFilter videofilter;
  RecurNN *net;
  int current_frame;
  float learn_rate;
  int osdebug;
  int playing;
  RnncaFrame *frame_prev;
  RnncaFrame *frame_now;
  RnncaFrame *play_frame;
  RecurNN **constructors;
  RnncaTrainer *trainers;
  RecurNN **train_nets;
  char *net_filename;
  int n_trainers;
  int hidden_size;
  char *pending_logfile;
  int training;
  u8 *training_map;
};

struct _GstRnncaClass
{
  GstVideoFilterClass parent_class;
};

#define GST_TYPE_RNNCA (gst_rnnca_get_type())
#define GST_RNNCA(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_RNNCA,GstRnnca))
#define GST_RNNCA_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_RNNCA,GstRnncaClass))
#define GST_IS_RNNCA(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_RNNCA))
#define GST_IS_RNNCA_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_RNNCA))

GType gst_rnnca_get_type(void);


G_END_DECLS
#endif /* __GOT_RNNCA_H__ */
