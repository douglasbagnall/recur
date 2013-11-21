#ifndef __GOT_RNNCA_H__
#define __GOT_RNNCA_H__

#include <gst/video/gstvideofilter.h>
#include "recur-common.h"
#include "recur-nn.h"
#include "badmaths.h"

G_BEGIN_DECLS

#define TRY_RELOAD 1
#define RNNCA_BIAS 0
#define RNNCA_N_TRAINERS 30
#define RNNCA_WIDTH 144
#define RNNCA_HEIGHT 96

#define RNNCA_BATCH_SIZE 1
#define RNNCA_RNG_SEED 11
#define RNNCA_BPTT_DEPTH 10
#define MOMENTUM 0.95
#define MOMENTUM_WEIGHT 0.5

#if RNNCA_BIAS
#define RNNCA_RNN_FLAGS (RNN_NET_FLAG_STANDARD | RNN_COND_USE_RAND |\
      RNN_COND_USE_SCALE | RNN_COND_USE_TALL_POPPY)
#else
#define RNNCA_RNN_FLAGS (RNN_NET_FLAG_NO_BIAS | RNN_COND_USE_RAND |\
       RNN_COND_USE_SCALE | RNN_COND_USE_TALL_POPPY)
#endif

#define PERIODIC_PGM_DUMP 511
#define PERIODIC_SAVE_NET 255

#define PERIODIC_CHECK_STASIS 0
#define PERIODIC_SHUFFLE_TRAINERS 15
#define PGM_DUMP_CHANGED_MASK 0

#define ARRAY_LEN(x) sizeof(x) / sizeof((x)[0])

const int RNNCA_YUV_OFFSETS[] = {
  -1, -1,   0, -1,   1, -1,
  -1,  0,            1,  0,
  -1,  1,   0,  1,   1,  1
};
const int RNNCA_YUV_LEN = ARRAY_LEN(RNNCA_YUV_OFFSETS);

#define USE_UV_ONLY_OFFSETS 1
#define USE_Y_ONLY_OFFSETS 1
#define USE_Y_MEAN_3_OFFSETS 0

#if USE_UV_ONLY_OFFSETS
const int RNNCA_UV_ONLY_OFFSETS[] = {

  -2, -2,          2, -2,
           0,  0,
  -2,  2,          2,  2,
  /*
       -1, -2,  1, -2,
  -2, -1,            2, -1,
             0,0,
  -2,  1,            2,  1,
       -1,  2,  1,  2
  */
};
const int RNNCA_UV_ONLY_LEN = ARRAY_LEN(RNNCA_UV_ONLY_OFFSETS);
#else
const int RNNCA_UV_ONLY_LEN = 0;
#endif


#if USE_Y_ONLY_OFFSETS
const int RNNCA_Y_ONLY_OFFSETS[] = {

       -1, -2,  1, -2,
  -2, -1,            2, -1,
    //       0,0,
  -2,  1,            2,  1,
       -1,  2,  1,  2

  /*
  -2, -2,          2, -2,
           0,  0,
  -2,  2,          2,  2,
  */

  /*
  -2, -2,  0, -2,  2, -2,
  -2,  0,          2,  0,
  -2,  2,  0,  2,  2,  2,
  */
};
const int RNNCA_Y_ONLY_LEN = ARRAY_LEN(RNNCA_Y_ONLY_OFFSETS);
#else
const int RNNCA_Y_ONLY_LEN = 0;
#endif

#if USE_Y_MEAN_3_OFFSETS
const int RNNCA_Y_MEAN_3_OFFSETS[] = {
  -1, -2,  0, -2, 1, -2,
  -1,  2,  0,  2, 1,  2,
  -2, -1, -2,  0, -2, 1,
   2, -1,  2,  0,  2, 1,
};
const int RNNCA_Y_MEAN_3_LEN = ARRAY_LEN(RNNCA_Y_MEAN_3_OFFSETS);
#else
const int RNNCA_Y_MEAN_3_LEN = 0;
#endif



#define RNNCA_N_FEATURES (((RNNCA_YUV_LEN * 3 + RNNCA_Y_ONLY_LEN \
              + RNNCA_UV_ONLY_LEN + (RNNCA_Y_MEAN_3_LEN / 3)) >> 1) + 2)

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
  float pending_learn_rate;
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
  int edges;
  float momentum;
  int momentum_soft_start;
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
