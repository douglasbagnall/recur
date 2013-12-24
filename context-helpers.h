#ifndef HAVE_CONTEXT_HELPERS_H
#define HAVE_CONTEXT_HELPERS_H
#include "recur-context.h"
#include "badmaths.h"
#include <stdbool.h>

static inline void
dump_frame(GstVideoFrame *f){
  GstVideoInfo *vi = &f->info;
  GST_DEBUG("Dumping frame %p", f);
  GST_DEBUG("flags %d buffer %p meta %p id %d",
      f->flags, f->buffer, f->meta, f->id);
  GST_DEBUG("data %p %p %p",
      f->data[0], f->data[1], f->data[2]);

  GST_DEBUG("width %d height %d size %zu",
      vi->width, vi->height, vi->size);
  GST_DEBUG("width %d height %d size %zu",
      vi->width, vi->height, vi->size);
}

static inline void blank_frame(RecurContext *context, GstVideoFrame *dest){
  for (uint i = 0; i < GST_VIDEO_INFO_N_PLANES(&context->video_info); i++){
    GST_DEBUG("i is %d", i);
    u8 *data = GST_VIDEO_FRAME_PLANE_DATA(dest, i);
    uint size = GST_VIDEO_FRAME_COMP_STRIDE(dest, i) * \
      GST_VIDEO_FRAME_COMP_HEIGHT(dest, i);
    GST_DEBUG("plane data %p size %u", data, size);
    memset(data, i == 0 ? 60 : 127, size);
  }
}

#define OUTPUT_SCALE 4


static inline void
blit_thumbnail(RecurContext *context, GstVideoFrame *dest, int x_pos, int y_pos){
  /* write to dest frame */
  blank_frame(context, dest);
  RecurFrame *f = &context->frame_queue[context->fq_head];
  int scale[3] = {2, 1, 1};
  u8 *s = (u8*)f->Y;
  for (int i = 0; i < 3; i++){
    int stride = GST_VIDEO_FRAME_COMP_STRIDE(dest, i);
    u8 *plane = GST_VIDEO_FRAME_COMP_DATA(dest, i);
    u8 *d = plane + (y_pos * stride + x_pos) * scale[i];
    if (scale[i] == 1){
      for (int y = 0; y < RECUR_WORKING_HEIGHT; y++){
        memcpy(d, s, RECUR_WORKING_WIDTH);
        d += stride;
        s += RECUR_WORKING_WIDTH;
      }
    }
    else if (scale[i] == 2) {
      for (int y = 0; y < RECUR_WORKING_HEIGHT; y++){
        for (int x = 0; x < RECUR_WORKING_WIDTH; x++){
          d[2 * x] = d[2 * x + 1] = s[x];
        }
        memcpy(d + stride, d, RECUR_WORKING_WIDTH * 2);
        d += stride * 2;
        s += RECUR_WORKING_WIDTH;
      }
    }
  }
  GST_DEBUG(" added thumbnails");
}
#endif
