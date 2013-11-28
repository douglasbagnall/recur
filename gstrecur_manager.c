/* GStreamer
 * Copyright (C) <2013> Douglas Bagnall <douglas@halo.gen.nz>
 *
 */

#include "gstrecur_manager.h"
#include "gstrecur_video.h"
#include "gstrecur_audio.h"
#include "recur-common.h"

#include <string.h>
#include <math.h>

/* static_functions */
static void gst_recur_manager_class_init(GstRecurManagerClass *g_class);
static void gst_recur_manager_init(GstRecurManager *self);
static void gst_recur_manager_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_recur_manager_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);

static gboolean plugin_init(GstPlugin *plugin);

GST_DEBUG_CATEGORY_STATIC (recur_manager_debug);
#define GST_CAT_DEFAULT recur_manager_debug

/* GstRecurManager signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_OSDEBUG,
};

#define DEFAULT_PROP_OSDEBUG FALSE

#define gst_recur_manager_parent_class parent_class
G_DEFINE_TYPE (GstRecurManager, gst_recur_manager, GST_TYPE_BIN)

/* Clean up */
static void
gst_recur_manager_finalize (GObject * obj){
  GST_DEBUG("in gst_recur_manager_finalize!\n");
  //GstRecurManager *self = GST_RECUR_MANAGER(obj);
}

static void
gst_recur_manager_class_init (GstRecurManagerClass * g_class)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class = (GstElementClass *) g_class;

  GST_DEBUG_CATEGORY_INIT (recur_manager_debug, "recur_manager", RECUR_LOG_COLOUR,
      "recur manager");
  gobject_class = G_OBJECT_CLASS (g_class);

  gobject_class->set_property = gst_recur_manager_set_property;
  gobject_class->get_property = gst_recur_manager_get_property;
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_recur_manager_finalize);

  g_object_class_install_property (gobject_class, PROP_OSDEBUG,
      g_param_spec_boolean ("osdebug", "On-screen debug",
          "on-screen debugging display [off]",
          DEFAULT_PROP_OSDEBUG,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (gstelement_class,
      "Recur manager sub-element",
      "Filter/Manager",
      "Mangles manager",
      "Douglas Bagnall <douglas@halo.gen.nz>");

  GST_INFO("gst manager class init\n");
}


static void
connect_ghost_pad(GstBin *bin, GstElement *element, const char *orig_name,
    const char *new_name)
{
  GstPad *pad = gst_element_get_static_pad (element, orig_name);
  gst_element_add_pad (GST_ELEMENT(bin), gst_ghost_pad_new (new_name, pad));
  gst_object_unref (GST_OBJECT (pad));
}

static void
gst_recur_manager_init (GstRecurManager * self)
{
  GstElement *audio, *video;
  GstBin *bin = GST_BIN(self);
  audio = gst_element_factory_make("recur_audio", NULL);
  video = gst_element_factory_make("recur_video", NULL);
  gst_bin_add (bin, audio);
  gst_bin_add (bin, video);

  connect_ghost_pad(bin, audio, "sink", "audio-sink");
  connect_ghost_pad(bin, audio, "src", "audio-src");
  connect_ghost_pad(bin, video, "sink", "video-sink");
  connect_ghost_pad(bin, video, "src", "video-src");

  self->context = malloc_aligned_or_die(sizeof(RecurContext));
  recur_context_init(self->context);

  gst_recur_audio_register_context (GST_RECUR_AUDIO(audio), self->context);
  gst_recur_video_register_context (GST_RECUR_VIDEO(video), self->context);

  GST_INFO("gst recur_manager init\n");
}


static void
gst_recur_manager_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstRecurManager *self = GST_RECUR_MANAGER (object);
  GST_DEBUG("gst_recur_manager_set_property\n");
  if (value){
    switch (prop_id) {
    case PROP_OSDEBUG:
      self->context->osdebug = g_value_get_boolean(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
    }
  }
}

static void
gst_recur_manager_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstRecurManager *self = GST_RECUR_MANAGER (object);
  switch (prop_id) {
    case PROP_OSDEBUG:
      g_value_set_boolean(value, self->context->osdebug);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}



static gboolean
plugin_init (GstPlugin * plugin)
{
  //GST_DEBUG_CATEGORY_INIT (recur_manager_debug, "recur_manager", RECUR_LOG_COLOUR, "recur_manager");
  GST_INFO("recur plugin init\n");
  gboolean manager = gst_element_register (plugin, "recur_manager", GST_RANK_NONE,\
      GST_TYPE_RECUR_MANAGER);
  gboolean video = gst_element_register (plugin, "recur_video", GST_RANK_NONE,\
      GST_TYPE_RECUR_VIDEO);
  gboolean audio = gst_element_register (plugin, "recur_audio", GST_RANK_NONE,\
      GST_TYPE_RECUR_AUDIO);
  return manager && video && audio;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    recur,
    "Add recur_managers to manager streams",
    plugin_init, VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
