/* GStreamer
 * Copyright (C) <2013> Douglas Bagnall <douglas@halo.gen.nz>
 *
 */

#include "gstrnnca.h"
#include "recur-common.h"
#include <gst/video/gstvideofilter.h>
#include <gst/video/video.h>

#include <string.h>
#include <math.h>

GST_DEBUG_CATEGORY_STATIC (rnnca_video_debug);
#define GST_CAT_DEFAULT rnnca_video_debug

/* GstRnncaVideo signals and args */
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
/* plugin_init    - registers plugin (once)
   XXX_base_init  - for the gobject class (once, obsolete)
   XXX_class_init - for global state (once)
   XXX_init       - for each plugin instance
*/
static void gst_rnnca_video_class_init(GstRnncaVideoClass *g_class);
static void gst_rnnca_video_init(GstRnncaVideo *self);
static void gst_rnnca_video_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_rnnca_video_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static GstFlowReturn gst_rnnca_video_transform_frame_ip(GstVideoFilter *base, GstVideoFrame *buf);

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

#define gst_rnnca_video_parent_class parent_class
G_DEFINE_TYPE (GstRnncaVideo, gst_rnnca_video, GST_TYPE_VIDEO_FILTER);

/* Clean up */
static void
gst_rnnca_video_finalize (GObject * obj){
  GST_DEBUG("in gst_rnnca_video_finalize!\n");
  //GstRnncaVideo *self = GST_RNNCA_VIDEO(obj);
}

static void
gst_rnnca_video_class_init (GstRnncaVideoClass * g_class)
{
  //GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS (g_class);
  GstElementClass *gstelement_class = (GstElementClass *) g_class;

  GST_DEBUG_CATEGORY_INIT (rnnca_video_debug, "rnnca_video", RECUR_LOG_COLOUR,
      "rnnca video");

  GObjectClass *gobject_class = G_OBJECT_CLASS (g_class);
  GstVideoFilterClass *vf_class = GST_VIDEO_FILTER_CLASS (g_class);

  gobject_class->set_property = gst_rnnca_video_set_property;
  gobject_class->get_property = gst_rnnca_video_get_property;
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_rnnca_video_finalize);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));

  gst_element_class_set_static_metadata (gstelement_class,
      "Rnnca video sub-element",
      "Filter/Video",
      "Mangles video",
      "Douglas Bagnall <douglas@halo.gen.nz>");

  vf_class->transform_frame_ip = GST_DEBUG_FUNCPTR (gst_rnnca_video_transform_frame_ip);
  vf_class->set_info = GST_DEBUG_FUNCPTR (set_info);
  GST_INFO("gst class init\n");
}

static void
gst_rnnca_video_init (GstRnncaVideo * self)
{
  //gst_element_create_all_pads(GST_ELEMENT(self));
  GST_INFO("gst rnnca_video init\n");
}



static gboolean
set_info (GstVideoFilter *filter,
    GstCaps *incaps, GstVideoInfo *in_info,
    GstCaps *outcaps, GstVideoInfo *out_info)
{
  GstRnncaVideo *self = GST_RNNCA_VIDEO (filter);
  self->video_info = *in_info;
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
gst_rnnca_video_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  //GstRnncaVideo *self = GST_RNNCA_VIDEO (object);
  GST_DEBUG("gst_rnnca_video_set_property\n");
  if (value){
    switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
    }
  }
}

static void
gst_rnnca_video_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  //GstRnncaVideo *self = GST_RNNCA_VIDEO (object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static inline void
remember_frame(GstRnncaVideo *self, GstVideoFrame *frame){

}

static inline void
maybe_learn(GstRnncaVideo *self){

}

static inline void
fill_frame(GstRnncaVideo *self, GstVideoFrame *frame){

}




static GstFlowReturn
gst_rnnca_video_transform_frame_ip(GstVideoFilter *filter,
    GstVideoFrame *frame)
{
  GstRnncaVideo *self = GST_RNNCA_VIDEO(filter);
  GstFlowReturn ret = GST_FLOW_OK;
  if (self->training){
    remember_frame(self, frame);
    maybe_learn(self);
  }
  if (self->playing){
    fill_frame(self, frame);
  }
  GST_LOG("rnnca_video_transform returning %d", ret);
  return ret;
}
