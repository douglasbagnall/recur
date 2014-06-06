//#include "recur-nn.h"
#include "recur-rng.h"
#include "pgm_dump.h"
#include <math.h>
#include <stdio.h>
#include <string.h>


static inline void
bernoulli_jumps(rand_ctx *rng, int *jumps, int len, double p){
  int i;
  double recip_lognp = 1.0 / log(1.0 - p);
  for (i = 0; i < len; i++){
    jumps[i] = ceil(log(1.0 - rand_double(rng)) * recip_lognp);
  }
}

static inline void
bernoulli_jumps_f(rand_ctx *rng, int *jumps, int len, float p){
  int i;
  float recip_lognp = 1.0f / logf(1.0f - p);
  float r[len];
  randomise_float_array(rng, r, len);
  for (i = 0; i < len; i++){
    jumps[i] = ceilf(logf(1.0f - r[i]) * recip_lognp);
  }
}



#define WIDTH 800
#define HEIGHT 600

#define N_BINS 100

int
main(void){
  rand_ctx rng;
  init_rand64(&rng, 2);
  u8 * im = malloc_aligned_or_die(WIDTH * HEIGHT);
  memset(im, 0, WIDTH * HEIGHT);
  int jumps[WIDTH];
  double rate = 0.25;
  for (int y = 0; y < HEIGHT; y++){
    for (int i = 0; i < 80 / rate; i++){
      bernoulli_jumps_f(&rng, jumps, WIDTH, rate);
      for (int x = *jumps - 1, j = 1; x < WIDTH && j < WIDTH; j++){
        if (im[y * WIDTH + x] < 254)
          im[y * WIDTH + x] += 2;
        else
          im[y * WIDTH + x] = 255;
        x += jumps[j];
      }
    }
    rate *= 0.99;
  }
  int bins[N_BINS];
  memset(bins, 0, sizeof(bins));
  for (int i = 0; i < WIDTH; i++){
    int j = jumps[i];
    //fprintf(stderr, "%i ", j);
    if (j < N_BINS)
      bins[j]++;
  }
  int scale = 1;
  int max = 0;
  for (int i = 0; i < N_BINS; i++){
    scale = MAX(scale, bins[i] / 90);
    if (bins[i])
      max = i;
  }
  for (int i = 0; i <= max; i++){
    fprintf(stderr, "\n%3d ", i);
    for (int j = 0; j < bins[i]; j+= scale){
      fputc('+', stderr);
    }
  }
  fputc('\n', stderr);
  pgm_dump(im, WIDTH, HEIGHT, "poisson.pgm");
}
