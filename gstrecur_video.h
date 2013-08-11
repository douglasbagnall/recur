#ifndef __GST_VIDEO_RECUR_VIDEO_H__
#define __GST_VIDEO_RECUR_VIDEO_H__

#include <gst/video/gstvideofilter.h>
#include "recur-context.h"

G_BEGIN_DECLS
#define GST_TYPE_RECUR_VIDEO \
  (gst_recur_video_get_type())
#define GST_RECUR_VIDEO(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_RECUR_VIDEO,GstRecurVideo))
#define GST_RECUR_VIDEO_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_RECUR_VIDEO,GstRecurVideoClass))
#define GST_IS_RECUR_VIDEO(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_RECUR_VIDEO))
#define GST_IS_RECUR_VIDEO_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_RECUR_VIDEO))


typedef struct _GstRecurVideo GstRecurVideo;
typedef struct _GstRecurVideoClass GstRecurVideoClass;

struct _GstRecurVideo
{
  GstVideoFilter videofilter;
  RecurContext *context;
};

struct _GstRecurVideoClass
{
  GstVideoFilterClass parent_class;
};

GType gst_recur_video_get_type(void);

void gst_recur_video_register_context (GstRecurVideo * self, RecurContext *context);

G_END_DECLS
#endif /* __GST_VIDEO_RECUR_VIDEO_H__ */
