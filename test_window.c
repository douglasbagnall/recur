#include "recur-common.h"
#include <math.h>
#include "badmaths.h"
#include <stdio.h>
#include <fenv.h>
#include <ctype.h>
#include "mdct.h"
#include "mfcc.h"
#include "window.h"

#define WINDOW_BITS 9
#define WINDOW_SIZE (1 << WINDOW_BITS)
#define WINDOW_NO (WINDOW_BITS - 6)


static inline void
compare_vorbis_window(int window_bits){
  int i;
  int window_size = 1 << window_bits;
  int window_no = window_bits - 6;
  int half_window = window_size / 2;

  float *recur_window = malloc_aligned_or_die((window_size + 2) * sizeof(float));
  recur_window_init(recur_window, window_size, RECUR_WINDOW_VORBIS, 1.0f);

  const float *vorbis_window = _vorbis_window_get(window_no);

  printf("window %d bits %d\n", window_bits, window_size);
  float worst_r = 0;
  float worst_rr = 0;
  for (i = 0; i < half_window; i++){
    float v = vorbis_window[i];
    float r = recur_window[i];
    float rr = recur_window[window_size - 1 - i];
    //printf("%3d vorbis %.3g mfcc %.3g %.3g diff %.3g %.3g\n",
    //    i, v, r, rr, v - r, v - rr);
    worst_r = MAX(fabsf(v - r), worst_r);
    worst_rr = MAX(fabsf(v - rr), worst_rr);
  }
  printf("worst diff: left %g right %g\n\n", worst_r, worst_rr);
  free(recur_window);
}



int
main(void){
  int i;
  for (i = 6; i < 12; i++){
    compare_vorbis_window(i);
  }
}
