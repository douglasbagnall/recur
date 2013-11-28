/* GStreamer
 * Copyright (C) <2013> Douglas Bagnall <douglas@halo.gen.nz>
 *
 */

#include "gstrecur_video.h"
#include "recur-common.h"
#include <gst/video/gstvideofilter.h>
#include <gst/video/video.h>

#include <string.h>
#include <math.h>

GST_DEBUG_CATEGORY_STATIC (recur_video_debug);
#define GST_CAT_DEFAULT recur_video_debug

/* GstRecurVideo signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
};


/* static_functions */
static void gst_recur_video_class_init(GstRecurVideoClass *g_class);
static void gst_recur_video_init(GstRecurVideo *self);
static void gst_recur_video_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_recur_video_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static GstFlowReturn gst_recur_video_transform_frame(GstVideoFilter *base, GstVideoFrame *inbuf, GstVideoFrame *outbuf);

static gboolean set_info (GstVideoFilter *filter,
    GstCaps *incaps, GstVideoInfo *in_info,
    GstCaps *outcaps, GstVideoInfo *out_info);

#define VIDEO_FORMATS " { I420 } "

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE (VIDEO_FORMATS))
    );

static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE (VIDEO_FORMATS))
    );

#define gst_recur_video_parent_class parent_class
G_DEFINE_TYPE (GstRecurVideo, gst_recur_video, GST_TYPE_VIDEO_FILTER)

/* Clean up */
static void
gst_recur_video_finalize (GObject * obj){
  GST_DEBUG("in gst_recur_video_finalize!\n");
  //GstRecurVideo *self = GST_RECUR_VIDEO(obj);
}

static void
gst_recur_video_class_init (GstRecurVideoClass * g_class)
{
  //GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS (g_class);
  GstElementClass *gstelement_class = (GstElementClass *) g_class;

  GST_DEBUG_CATEGORY_INIT (recur_video_debug, "recur_video", RECUR_LOG_COLOUR,
      "recur video");

  GObjectClass *gobject_class = G_OBJECT_CLASS (g_class);
  GstVideoFilterClass *vf_class = GST_VIDEO_FILTER_CLASS (g_class);

  gobject_class->set_property = gst_recur_video_set_property;
  gobject_class->get_property = gst_recur_video_get_property;
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_recur_video_finalize);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));

  gst_element_class_set_static_metadata (gstelement_class,
      "Recur video sub-element",
      "Filter/Video",
      "Mangles video",
      "Douglas Bagnall <douglas@halo.gen.nz>");

  vf_class->transform_frame = GST_DEBUG_FUNCPTR (gst_recur_video_transform_frame);
  vf_class->set_info = GST_DEBUG_FUNCPTR (set_info);
  GST_INFO("gst class init\n");
}

static void
gst_recur_video_init (GstRecurVideo * self)
{
  //gst_element_create_all_pads(GST_ELEMENT(self));
  GST_INFO("gst recur_video init\n");
}


static gboolean
set_info (GstVideoFilter *filter,
    GstCaps *incaps, GstVideoInfo *in_info,
    GstCaps *outcaps, GstVideoInfo *out_info)
{
  GstRecurVideo *self = GST_RECUR_VIDEO (filter);
  recur_context_set_video_properties(self->context, in_info);
  return TRUE;
}


static inline void
set_string_prop(const GValue *value, const char **target){
  const char *s = g_value_dup_string(value);
  size_t len = strlen(s);
  if(len){
    *target = s;
  }
}

static void
gst_recur_video_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  //GstRecurVideo *self = GST_RECUR_VIDEO (object);
  GST_DEBUG("gst_recur_video_set_property\n");
  if (value){
    switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
    }
  }
}

static void
gst_recur_video_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  //GstRecurVideo *self = GST_RECUR_VIDEO (object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


static GstFlowReturn
gst_recur_video_transform_frame (GstVideoFilter *filter,
    GstVideoFrame *inframe, GstVideoFrame *outframe)
{
  GstRecurVideo *self = GST_RECUR_VIDEO(filter);
  recur_queue_video_buffer(self->context, inframe->buffer);
  recur_fill_video_frame(self->context, outframe);
  GST_LOG("recur_video_transform returning OK");
  return GST_FLOW_OK;
}

void
gst_recur_video_register_context (GstRecurVideo * self, RecurContext *context)
{
  self->context = context;
  GST_INFO("video_register_context\n");
}
