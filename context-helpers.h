#include "recur-context.h"
#include "badmaths.h"
#include <stdbool.h>
#ifndef HAVE_CONTEXT_HELPERS_H
#define HAVE_CONTEXT_HELPERS_H

static inline void
dump_frame(GstVideoFrame *f){
  GstVideoInfo *vi = &f->info;
  GST_DEBUG("Dumping frame %p", f);
  GST_DEBUG("flags %d buffer %p meta %p id %d",
      f->flags, f->buffer, f->meta, f->id);
  GST_DEBUG("data %p %p %p",
      f->data[0], f->data[1], f->data[2]);
  GST_DEBUG("map %p %p %p",
      f->map[0], f->map[1], f->map[2]);

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

static inline void
blit_planar_u8(GstVideoFrame *dest, u8 *planes,
    int x_pos, int y_pos, int width, int height, int scale){
  int plane_scale[3] = {2, 1, 1};
  u8*s = planes;
  for (int i = 0; i < 3; i++){
    int pscale = scale * plane_scale[i];
    int stride = GST_VIDEO_FRAME_COMP_STRIDE(dest, i);
    u8 *plane = GST_VIDEO_FRAME_COMP_DATA(dest, i);
    u8 *d = plane + (y_pos * stride + x_pos) * plane_scale[i];
    if (pscale == 1){
      for (int y = 0; y < height; y++){
        memcpy(d, s, width);
        d += stride;
        s += width;
      }
    }
    else if (pscale == 2) {
      for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
          d[2 * x] = d[2 * x + 1] = s[x];
        }
        memcpy(d + stride, d, width * 2);
        d += stride * 2;
        s += width;
      }
    }
    else if (pscale == 4) {
      for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
          d[4 * x] = d[4 * x + 1] = d[4 * x + 2] = d[4 * x + 3] = s[x];
        }
        memcpy(d + stride, d, width * 4);
        memcpy(d + stride * 2, d, width * 4);
        memcpy(d + stride * 3, d, width * 4);
        d += stride * 4;
        s += width;
      }
    }
    else {
      for (int y = 0; y < height; y++){
        for (int x = 0; x < width; x++){
          for (int k = 0; k < pscale; k++){
            d[pscale * x + k] = s[x];
          }
        }
        for (int k = 1; k < pscale; k++){
          memcpy(d + stride * k, d, width * pscale);
        }
        d += stride * pscale;
        s += width;
      }
    }
  }
  GST_DEBUG(" added planar u8 something");
}

static inline void
blit_planar_float(GstVideoFrame *dest, const float *planes,
    int x_pos, int y_pos, int width, int height, int scale,
    bool sigmoid_norm){
  int len = width * height * 3;
  u8 bytes[len];
  if (sigmoid_norm){
    fast_sigmoid_byte_array(bytes, planes, len);
  }
  else {
    for (int i = 0; i < len; i++){
      bytes[i] = planes[i] * 255.99f;
    }
  }
  blit_planar_u8(dest, bytes, x_pos, y_pos, width, height, scale);
}


static inline void
stretch_row(const u8 *restrict src, u8 *restrict dest,
    const int s_width, const int d_width)
{
  int i = 0;
  int j = 0;
  int k = 0;
  for (; i < d_width; i++, j+= s_width){
    if (j > d_width){

      j -= d_width;
      k++;
    }
    dest[i] = src[k];
  }
}


static inline void
fill_from_planar_u8(GstVideoFrame *frame, const u8 *restrict src,
    int s_width, int s_height){
  for (int i = 0; i < 3; i++){
    int stride = GST_VIDEO_FRAME_COMP_STRIDE(frame, i);
    int d_width = GST_VIDEO_FRAME_COMP_WIDTH(frame, i);
    int d_height = GST_VIDEO_FRAME_COMP_HEIGHT(frame, i);
    u8 *plane = GST_VIDEO_FRAME_COMP_DATA(frame, i);
    int i = 0;
    int j = 0;
    int k = 0;
    stretch_row(src, plane, s_width, d_width);
    u8 *current_row = plane;
    for (; i < d_height; i++, j+= s_height){
      if (j > d_height){
        j -= d_height;
        k++;
        current_row = plane + i * stride;
        stretch_row(src + k * s_width, current_row, s_width, d_width);

      }
      else {
        memcpy(plane + i * stride, current_row, d_width);
      }
    }
    src += s_width * s_height;
  }
}

#endif
