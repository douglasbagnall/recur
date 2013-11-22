#ifndef HAVE_PLAYER_COMMON_H
#define HAVE_PLAYER_COMMON_H 1

#define URI_PREFIX "file://" TEST_VIDEO_DIR "/"

#include <gst/video/videooverlay.h>
#include <gst/gst.h>
#include "recur-common.h"
#include <gtk/gtk.h>
#include <gdk/gdkx.h>
#include "path.h"

static inline void
toggle_fullscreen(GtkWidget *widget){
  GdkWindow *gdk_window = gtk_widget_get_window(widget);
  GdkWindowState state = gdk_window_get_state(gdk_window);
  if (state & GDK_WINDOW_STATE_FULLSCREEN){
    gtk_window_unfullscreen(GTK_WINDOW(widget));
  }
  else{
    gtk_window_fullscreen(GTK_WINDOW(widget));
  }
}

static inline void
hide_mouse(GtkWidget *widget){
  GdkWindow *w = gtk_widget_get_window(widget);
  GdkDisplay *display = gdk_display_get_default();
  GdkCursor *cursor = gdk_cursor_new_for_display(display, GDK_BLANK_CURSOR);
  gdk_window_set_cursor(w, cursor);
  g_object_unref (cursor);
}

static inline void
set_up_loop(GstElement *source, int flags){
  DEBUG("loooooping\n");
  if (! gst_element_seek(source, 1.0, GST_FORMAT_TIME,
          flags,
          GST_SEEK_TYPE_SET, 0,
          GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE
      )){
    GST_WARNING("Seek failed!\n");
  }
}

static guintptr video_window_handle = 0;

static GstBusSyncReply
sync_bus_call(GstBus *bus, GstMessage *msg, gpointer data)
{
 if (!gst_is_video_overlay_prepare_window_handle_message (msg))
   return GST_BUS_PASS;

 if (video_window_handle != 0) {
   GstVideoOverlay *overlay;
   // GST_MESSAGE_SRC (msg) will be the video sink element
   overlay = GST_VIDEO_OVERLAY (GST_MESSAGE_SRC (msg));
   gst_video_overlay_set_window_handle (overlay, video_window_handle);
 }
 else {
   g_warning ("Should have obtained video_window_handle by now!");
 }
 gst_message_unref (msg);
 return GST_BUS_DROP;
}

#endif
