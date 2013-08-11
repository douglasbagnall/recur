/* GStreamer
 * Copyright (C) <2013> Douglas Bagnall <douglas@halo.gen.nz>
 *
 */

#include "gstrecur_audio.h"
#include <string.h>
#include <math.h>

GST_DEBUG_CATEGORY_STATIC (recur_audio_debug);
#define GST_CAT_DEFAULT recur_audio_debug

/* GstRecurAudio signals and args */
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
static void gst_recur_audio_class_init(GstRecurAudioClass *g_class);
static void gst_recur_audio_init(GstRecurAudio *self);
static void gst_recur_audio_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_recur_audio_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static GstFlowReturn gst_recur_audio_transform(GstBaseTransform *base, GstBuffer *inbuf, GstBuffer *outbuf);
static gboolean gst_recur_audio_setup(GstAudioFilter * filter, const GstAudioInfo * info);

#define gst_recur_audio_parent_class parent_class
G_DEFINE_TYPE (GstRecurAudio, gst_recur_audio, GST_TYPE_AUDIO_FILTER);

/* Clean up */
static void
gst_recur_audio_finalize (GObject * obj){
  GST_DEBUG("in gst_recur_audio_finalize!\n");
  //GstRecurAudio *self = GST_RECUR_AUDIO(obj);
}

static void
gst_recur_audio_class_init (GstRecurAudioClass * klass)
{
  GST_DEBUG_CATEGORY_INIT (recur_audio_debug, "recur_audio", RECUR_LOG_COLOUR,
      "recur audio");
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS (klass);
  GstAudioFilterClass *af_class = GST_AUDIO_FILTER_CLASS (klass);
  gobject_class->set_property = gst_recur_audio_set_property;
  gobject_class->get_property = gst_recur_audio_get_property;
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_recur_audio_finalize);
  /*16kHz, single channel, 16 bit signed little endian PCM*/
  GstCaps *caps = gst_caps_new_simple ("audio/x-raw",
     "format", G_TYPE_STRING, RECUR_AUDIO_FORMAT,
     "rate", G_TYPE_INT, RECUR_AUDIO_RATE,
     "channels", G_TYPE_INT, RECUR_AUDIO_CHANNELS,
     NULL);

  gst_audio_filter_class_add_pad_templates (af_class, caps);

  gst_element_class_set_static_metadata (element_class,
      "Recur audio sub-element",
      "Filter/Audio",
      "Mangles audio",
      "Douglas Bagnall <douglas@halo.gen.nz>");

  trans_class->transform = GST_DEBUG_FUNCPTR (gst_recur_audio_transform);
  af_class->setup = GST_DEBUG_FUNCPTR (gst_recur_audio_setup);
  GST_INFO("gst audio class init\n");
}

static void
gst_recur_audio_init (GstRecurAudio * self)
{
  GST_INFO("gst recur_audio init\n");
}

static gboolean
gst_recur_audio_setup(GstAudioFilter * base, const GstAudioInfo * info){
  GST_INFO("gst audio setup\n");
  GstRecurAudio *self = GST_RECUR_AUDIO(base);
  self->context->audio_info = *info;
  GST_DEBUG_OBJECT (self,
      "info: %" GST_PTR_FORMAT, info);
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
gst_recur_audio_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  //GstRecurAudio *self = GST_RECUR_AUDIO (object);
  GST_DEBUG("gst_recur_audio_set_property\n");
  if (value){
    switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
    }
  }
}

static void
gst_recur_audio_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  //GstRecurAudio *self = GST_RECUR_AUDIO (object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}


static GstFlowReturn
gst_recur_audio_transform (GstBaseTransform * base, GstBuffer *inbuf, GstBuffer *outbuf)
{
  GstRecurAudio *self = GST_RECUR_AUDIO(base);
  GstFlowReturn ret = GST_FLOW_OK;
  recur_queue_audio_segment(self->context, inbuf);
  recur_fill_audio_segment(self->context, outbuf);
  GST_LOG("recur_audio_transform returning OK");
  //exit:
  return ret;
}

void
gst_recur_audio_register_context (GstRecurAudio * self, RecurContext *context)
{
  self->context = context;
  GST_INFO("audio_register_context\n");
}
