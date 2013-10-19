#ifndef __GST_VIDEO_RNNCA_VIDEO_H__
#define __GST_VIDEO_RNNCA_VIDEO_H__

#include <gst/video/gstvideofilter.h>
#include "recur-common.h"
#include "recur-nn.h"
#include "badmaths.h"

G_BEGIN_DECLS
#define GST_TYPE_RNNCA_VIDEO \
  (gst_rnnca_video_get_type())
#define GST_RNNCA_VIDEO(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_RNNCA_VIDEO,GstRnncaVideo))
#define GST_RNNCA_VIDEO_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_RNNCA_VIDEO,GstRnncaVideoClass))
#define GST_IS_RNNCA_VIDEO(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_RNNCA_VIDEO))
#define GST_IS_RNNCA_VIDEO_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_RNNCA_VIDEO))

#define RNNCA_N_TRAINERS 50
#define RNNCA_WIDTH 128
#define RNNCA_HEIGHT 96

typedef struct _RnncaTrainer {
  RecurNN *net;
  int x;
  int y;
} RnncaTrainer;


typedef struct _GstRnncaVideo GstRnncaVideo;
typedef struct _GstRnncaVideoClass GstRnncaVideoClass;

struct _GstRnncaVideo
{
  GstVideoFilter videofilter;
  GstVideoInfo video_info;
  RecurNN *net;
  int current_frame;
  float learn_rate;
  int osdebug;
  u8 *planes;
  int playing;
  int training;
  RecurNN *constructors[RNNCA_WIDTH * RNNCA_HEIGHT];
  RnncaTrainer trainers[RNNCA_N_TRAINERS];
};

struct _GstRnncaVideoClass
{
  GstVideoFilterClass parent_class;
};

GType gst_rnnca_video_get_type(void);


G_END_DECLS
#endif /* __GST_VIDEO_RNNCA_VIDEO_H__ */
