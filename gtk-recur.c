/*Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL

This is a gtk app that runs the recur plugin. Try --help.
*/

#include "player-common.h"

#define WIDTH  800
#define HEIGHT 600

#define VID_LAGOS "movies/louis-theroux-lagos/louis.theroux.law.and.disorder.in.lagos.ws.pdtv.xvid-waters.avi"

#define VID_TEARS "F30275.mov"

static const char *PIPELINE_TEMPLATE = ("uridecodebin name=src "
    " ! videoscale method=nearest-neighbour ! videoconvert"
    " ! video/x-raw, format=I420, width=" QUOTE(WIDTH) ", height=" QUOTE(HEIGHT)
    " ! recur_manager name=recur osdebug=0 ! videoconvert"
    " ! xvimagesink name=videosink force-aspect-ratio=false"
    " recur. ! audio/x-raw ! fakesink"
    " src. ! audioconvert ! audioresample ! recur.");

static gboolean option_fullscreen = FALSE;
static gint option_width = WIDTH;
static gint option_height = HEIGHT;
static char *option_uri = URI_PREFIX VID_LAGOS;

static GOptionEntry entries[] =
{
  { "full-screen", 'f', 0, G_OPTION_ARG_NONE, &option_fullscreen,
    "run full screen", NULL },
  { "width", 'w', 0, G_OPTION_ARG_INT, &option_width, "width of each screen", NULL },
  { "height", 'h', 0, G_OPTION_ARG_INT, &option_height, "height of screen", NULL },
  { "uri", 'u', 0, G_OPTION_ARG_FILENAME, &option_uri, "URI to play", NULL },
  { NULL, 0, 0, 0, NULL, NULL, NULL }
};

static gboolean
key_press_event_cb(GtkWidget *widget, GdkEventKey *event, gpointer data)
{
  switch (event->keyval){
  case 'f':
    toggle_fullscreen(widget);
    break;
  case 'q':
    g_signal_emit_by_name(widget, "destroy");
    break;
  default:
    break;
  }
  return TRUE;
}

static void
video_widget_realize_cb (GtkWidget * widget, gpointer data)
{
  gulong xid = GDK_WINDOW_XID (gtk_widget_get_window (widget));
  video_window_handle = xid;

  //static const GdkColor black = {0, 0, 0, 0};
  gtk_window_set_default_size(GTK_WINDOW(widget), option_width, option_height);
  hide_mouse(widget);
  if (option_fullscreen){
    gtk_window_fullscreen(GTK_WINDOW(widget));
  }
}

gboolean bus_callback(GstBus *bus, GstMessage *msg, GMainLoop *loop)
{
  switch (GST_MESSAGE_TYPE(msg))
    {
    case GST_MESSAGE_EOS:
      g_main_loop_quit(loop);
      gtk_main_quit();
      break;
    default:
      break;
    }
  return TRUE;
}

static void
src_drained_cb(GstElement *src, GMainLoop *loop)
{
  g_main_loop_quit(loop);
  gtk_main_quit();
}


static GstElement *
make_pipeline(GError **parse_error,  GMainLoop *loop){
  GstElement *pipeline = gst_parse_launch(PIPELINE_TEMPLATE,
      parse_error);
  GstElement *src =  gst_bin_get_by_name(GST_BIN(pipeline), "src");
  g_object_set(G_OBJECT(src),
      "uri", option_uri,
      NULL);
  DEBUG("uri is %s", option_uri);

  g_signal_connect(src, "drained",
      G_CALLBACK(src_drained_cb), loop);

  return pipeline;
}

static void
options(int argc, char *argv[]){
  GOptionGroup *gst_opts = gst_init_get_option_group();
  GOptionGroup *gtk_opts = gtk_get_option_group(TRUE);
  GOptionContext *ctx = g_option_context_new("...!");
  g_option_context_add_main_entries(ctx, entries, NULL);
  g_option_context_add_group(ctx, gst_opts);
  g_option_context_add_group(ctx, gtk_opts);
  GError *error = NULL;
  if (!g_option_context_parse(ctx, &argc, &argv, &error)){
    g_print ("Error initializing: %s\n", GST_STR_NULL(error->message));
    exit (1);
  }
  g_option_context_free(ctx);
}

void destroy_cb(GtkWidget * widget, GMainLoop *loop)
{
  g_main_loop_quit(loop);
  gtk_main_quit();
}

gint main (int argc, char *argv[])
{
  options(argc, argv);
  GMainLoop *loop = g_main_loop_new(NULL, FALSE);
  GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

  /*Tell gstreamer to look locally for the plugin*/
  GstRegistry *registry;
  registry = gst_registry_get();
  gst_registry_scan_path(registry, "plugins");

  GError *parse_error = NULL;
  GstElement *pipeline = make_pipeline(&parse_error, loop);
  DEBUG("pipeline is %p", pipeline);

  GstBus *bus = gst_pipeline_get_bus(GST_PIPELINE(pipeline));
  gst_bus_set_sync_handler(bus, (GstBusSyncHandler)sync_bus_call, NULL, NULL);
  gst_bus_add_watch(bus, (GstBusFunc)bus_callback, loop);
  gst_object_unref(bus);

  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  g_signal_connect(G_OBJECT(window), "key-press-event",
      G_CALLBACK(key_press_event_cb), NULL);

  DEBUG("pipeline is %p", pipeline);

  g_signal_connect(G_OBJECT(window), "destroy",
      G_CALLBACK(destroy_cb), loop);

  g_signal_connect(window, "realize",
    G_CALLBACK(video_widget_realize_cb), NULL);

  gtk_widget_show_all(window);
  //hide_mouse(window);

  g_main_loop_run(loop);

  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);
  return 0;
}
