#ifndef HAVE_RESCALE_H
#define HAVE_RESCALE_H

#include "recur-common.h"

struct _Image {
  uint width;
  uint height;
  u8 *data;
};

typedef struct _Image Image;

__attribute__((malloc)) Image *
recur_load_pgm_file(const char *filename);

void
recur_exact_downscale(const u8 *restrict src, const int s_width,
    const int s_height, const int s_stride,
    u8 *restrict dest, const int d_width,
    const int d_height, const int d_stride);


void
recur_skipping_downscale(const u8 *restrict src, const int s_width,
    const int s_height, const int s_stride,
    u8 *restrict dest, const int d_width,
    const int d_height, const int d_stride);

void
recur_adaptive_downscale(const u8 *src, const int s_width,
    const int s_height, const int s_stride,
    u8 *dest, const int d_width,
    const int d_height, const int d_stride);

void
recur_float_downscale(const float *restrict src, const int s_width,
    const int s_height, const int s_stride,
    float *restrict dest, const int d_width,
    const int d_height, const int d_stride);


/*recur_integer_downscale_to_float scales u8 planes down to smaller [0,1)
  float planes, using nearest-neighbour search (for now) */

static inline void
recur_integer_downscale_to_float(const u8 *im, float *dest, int stride,
    int left, int top, int w, int h, int scale){
  const u8 *plane = im + top * stride + left;
  int y, x;
  /*w, h are target width and height. width is assumed to be stride. */
  for (y = 0; y < h; y++){
    for (int y2 = 0; y2 < scale; y2++){
      for (x = 0; x < w; x++){
        for (int x2 = 0; x2 < scale; x2++){
          dest[y * w + x] += plane[(y * scale + y2) * stride + x * scale + x2];
        }
      }
    }
  }
  for (int i = 0; i < w * h; i++){
    dest[i] /= (scale * scale * 256.0);
  }
}

#endif
