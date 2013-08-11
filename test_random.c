#include "recur-rng.h"
#include "pgm_dump.h"
#include <stdio.h>
#include <string.h>


#define PLOT_RESOLUTION 100
#define GAUSSIAN_SAMPLES 50000

static inline int
scale_and_confine(float f, float scale, int centre, int min, int max){
  int i;
  f *= scale;
  f += centre + 0.5;
  i = (int)f;
  if (i < min)
    i = min;
  if (i > max)
    i = max;
  return i;
}

#define ITERATIONS 100000000

#define time_fn(x) sum_a = sum_b = 0.0;                    \
  START_TIMER(time_ ## x)                                  \
  for (i = 0; i < ITERATIONS; i++){                        \
    x(rng, &a, &b, 0.1f);                                  \
    sum_a += a;                                            \
    sum_b += b;                                                  \
  }                                                              \
  DEBUG_TIMER(time_ ## x);                                       \
  DEBUG("(%d iterations) %f %f", ITERATIONS, sum_a, sum_b);


static inline void
singlecheap(rand_ctx *rng, float *restrict a,
    float *restrict b, const float variance){
  *a = cheap_gaussian_noise(rng) * variance;
  *b = cheap_gaussian_noise(rng) * variance;
}

void test_gauss2(rand_ctx *rng){
  int i;
  float a, b;
  float sum_a, sum_b;
  time_fn(doublecheap_gaussian_noise_f);
  //time_fn(doublecheap_gaussian_noise);
  time_fn(doublecheap2_gaussian_noise_f);
  //time_fn(doublecheap2_gaussian_noise);
  //time_fn(doublecheap3_gaussian_noise_f);
  //time_fn(doublecheap3_gaussian_noise);
  time_fn(singlecheap);
}



void
test_gauss(rand_ctx *rng, int res, int samples){
  int i, x, y;
  float a, b;
  int len = res * 8;
  float zero = res * 4;
  int bins[len];
  memset(bins, 0, sizeof(bins));
  START_TIMER(t);
  for (i = 0; i < samples / 2; i++){
    doublecheap2_gaussian_noise_f(rng, &a, &b, 1.0f);
    //a = cheap_gaussian_noise(rng);
    //b = cheap_gaussian_noise(rng);
    bins[scale_and_confine(a, res, zero, 0, len - 1)]++;
    bins[scale_and_confine(b, res, zero, 0, len - 1)]++;
  }
  DEBUG_TIMER(t);

  int max = 0;
  for (i = 0; i < len; i++){
    if (bins[i] > max)
      max = bins[i];
  }
  u8 *plot = malloc((max + 1) * len);
  memset (plot, 0, (max + 1) * len);
  i = 0;
  for (y = max; y >= 0; y--){
    for (x = 0; x < len; x++, i++){
      if (bins[x] > y)
        plot[i] = 255;
    }
  }
#if 0
  for (x = 0; x < len; x++, i++){
    DEBUG("%i %i", x, bins[x]);
  }
#endif

  pgm_dump(plot, len, max, "gaussian.pgm");
  free(plot);
}


int
main(void){
  rand_ctx rng;
  init_rand64(&rng, 12345);
  test_gauss2(&rng);
  //test_gauss(&rng, PLOT_RESOLUTION, GAUSSIAN_SAMPLES);
}
