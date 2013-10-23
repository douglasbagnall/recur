#ifndef HAVE_BLIT_HELPERS_H
#define HAVE_BLIT_HELPERS_H
#include "badmaths.h"
#include <stdbool.h>

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
