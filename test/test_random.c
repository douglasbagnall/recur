#include "recur-rng.h"
#include "pgm_dump.h"
#include <stdio.h>
#include <string.h>


#define PLOT_RESOLUTION 100
#define OVERSAMPLE_BITS 12
#define GAUSSIAN_SAMPLES (50000 << OVERSAMPLE_BITS)

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

void time_fn(rand_ctx *rng, void (*fn)(rand_ctx *, float *, float *, float),
    const char *name){
  float var_a = 0;
  float var_b = 0;
  float mean_a = 0;
  float mean_b = 0;
  float a, b, d;
  START_TIMER(fn);
  for (int i = 1; i <= ITERATIONS; i++){
    fn(rng, &a, &b, 1e-7f);
    d = a - mean_a;
    mean_a += d / i;
    var_a += d * (a - mean_a);

    d = b - mean_b;
    mean_b += d / i;
    var_b += d * (b - mean_a);
  }
  var_a /= ITERATIONS;
  var_b /= ITERATIONS;
  DEBUG("%s %d iterations", name, ITERATIONS);
  DEBUG_TIMER(fn);
  DEBUG("mean   a %4e b %4e", mean_a, mean_b);
  DEBUG("stddev a %4e b %4e", sqrtf(var_a), sqrtf(var_b));
  //DEBUG("sum    a %4e b %4e", sum_a, sum_b);
}

static void
singlecheap(rand_ctx *rng, float *restrict a,
    float *restrict b, const float deviation){
  *a = cheap_gaussian_noise(rng) * deviation;
  *b = cheap_gaussian_noise(rng) * deviation;
}

void test_gauss2(rand_ctx *rng){
  time_fn(rng, doublecheap_gaussian_noise_f, "doublecheap");
  time_fn(rng, doublecheap2_gaussian_noise_f, "doublecheap2");
  time_fn(rng, singlecheap, "singlecheap");
}


static void
test_gauss(rand_ctx *rng, int res, int samples, int bits,
    void (*fn)(rand_ctx *, float *, float *, float),
    char *filename){
  int i, x, y;
  float a, b;
  int len = res * 8;
  float zero = res * 4;
  int bins[len];
  memset(bins, 0, sizeof(bins));
  for (i = 0; i < samples / 2; i++){
    fn(rng, &a, &b, 0.8f);
    bins[scale_and_confine(a, res, zero, 0, len - 1)]++;
    bins[scale_and_confine(b, res, zero, 0, len - 1)]++;
  }

  int max = 0;
  for (i = 0; i < len; i++){
    if (bins[i] > max)
      max = bins[i];
  }
  int height = (max >> bits) + 1;
  u8 *plot = malloc((height + 1) * len);
  memset (plot, 0, (height + 1) * len);
  i = 0;
  for (y = height; y >= 0; y--){
    for (x = 0; x < len; x++, i++){
      int h = bins[x] >> bits;
      if (h == y){
        plot[i] = (bins[x] >> (bits - 8)) & 255;
      }
      else if (h > y){
        plot[i] = 255;
      }
    }
  }
#if 0
  for (x = 0; x < len; x++, i++){
    DEBUG("%i %i", x, bins[x]);
  }
#endif

  pgm_dump(plot, len, height, filename);
  free(plot);
}


int
main(void){
  rand_ctx rng;
  init_rand64(&rng, 12345);
  //test_gauss2(&rng);
  test_gauss(&rng, PLOT_RESOLUTION, GAUSSIAN_SAMPLES, OVERSAMPLE_BITS,
      doublecheap_gaussian_noise_f,
      "doublecheap.pgm");
  test_gauss(&rng, PLOT_RESOLUTION, GAUSSIAN_SAMPLES, OVERSAMPLE_BITS,
      doublecheap2_gaussian_noise_f,
      "doublecheap2.pgm");
  test_gauss(&rng, PLOT_RESOLUTION, GAUSSIAN_SAMPLES, OVERSAMPLE_BITS,
      singlecheap, "singlecheap.pgm");
}
