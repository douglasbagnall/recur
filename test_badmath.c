#include "recur-nn.h"
#include "pgm_dump.h"
#include "badmaths.h"
#include <math.h>


/*suboptimal tanh's */

static inline float
pade_tanhf(float x)
{
  SYMMETRIC_EXTREMA_CLAMP(x, 3.0f, -1.0f, 1.0f);
  return x * (27.0f + x * x) / (27.0f + 9.0f * x * x);
}

static inline float
fast_exp_tanhf(float x)
{
  SYMMETRIC_EXTREMA_CLAMP(x, 7.0f, -1.0f, 1.0f);
  return 2.0f / (1.0f + fast_expf(-2.0f * x)) - 1.0f;
}

static inline float
lambert_76_tanhf(float x)
{
  /*based on
    http://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
  */
  SYMMETRIC_EXTREMA_CLAMP(x, 4.97f, -1.0f, 1.0f);
  float x2 = x * x;
  float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return a / b;
}

static inline float
lambert_56_tanhf(float x)
{
  //(21*x^5 + 1260*x^3 + 10395*x) /
  //(x^6 + 210*x^4 + 4725*x^2 + 10395)
  SYMMETRIC_EXTREMA_CLAMP(x, 4.97f, -1.0f, 1.0f);
  float x2 = x * x;
  float a = x * (10395.0f + x2 * (1260.0f + x2 * 21.0f));
  float b = 10395.0f + x2 * (4725.0f + x2 * (210.0f + x2));
  return a / b;
}

static inline float
best_rational_tanhf(float x)
{
  SYMMETRIC_EXTREMA_CLAMP(x, 5.0f, -1.0f, 1.0f);
  float a = 1.0f;
  if (x < 0){
    x = -x;
    a = -1.0f;
  }
  a *= (-.67436811832e-5+(.2468149110712040+(.583691066395175e-1+.3357335044280075e-1*x)*x)*x);
  float b = (.2464845986383725+(.609347197060491e-1 + (.1086202599228572 + .2874707922475963e-1 * x) * x) * x);
  return a / b;
}

static inline float
pade2_tanhf(float x)
{
  SYMMETRIC_EXTREMA_CLAMP(x, 3.0f, -1.0f, 1.0f);
  float x2 = x * x;
  float a = x * (10.0f + x2) * (60.0f + x2);
  float b = 600.0f + 270.0f * x2 + 11.0f * x2 * x2 + 0.16666666666667f * x2 * x2 * x2;
  return a / b;
}


/*****************************testing functions *************/

struct delta_stats {
  float delta;
  float delta2;
  float worst;
};

static inline struct delta_stats
test_float_range(float (*fn1)(float), float (*fn2)(float),
    float start, float stop, float step){
  struct delta_stats stats = {0, 0, 0};
  float count = 0.0;
  for (float f = start; f < stop; f += step){
    float a1 = fn1(f);
    float a2 = fn2(f);
    float d = (a1 - a2) / (a1 ? a1 : 1);
    //DEBUG("%5.3f fn1 %f fn2 %f diff %g", f, a1, a2, d);
    stats.delta2 += d * d;
    stats.delta += fabsf(d);
    if (fabsf(d) > stats.worst)
      stats.worst = fabsf(d);
    count ++;
  }
  stats.delta2 = sqrt(stats.delta2 / count);
  stats.delta = stats.delta / count;
  return stats;
}


#define DELTA_RANGE(fn1, fn2, start, stop, step)                        \
  do {                                                                  \
    struct delta_stats stats = test_float_range(fn1, fn2, start, stop, step); \
  DEBUG("%8s - %20s (%g %g %.3g) rms delta %f; delta %f; worst %f",    \
      QUOTE(fn1), QUOTE(fn2), start, stop, step,                        \
      stats.delta2, stats.delta, stats.worst);                          \
  } while(0)


#define TIME_RANGE(fn, start, stop, step)                \
  START_TIMER(fn);                                         \
  float sum ## fn = 0.0;                                   \
  for (float i = start; i < stop; i += step){              \
    sum ## fn += fn(i);                                    \
  }                                                        \
  DEBUG_TIMER(fn);



#define TIME_RAND(fn, N, scale)                        \
  init_rand64(&rng, 1);                                \
  START_TIMER(fn);                                     \
  float sum ## fn = 0.0;                               \
  for (int i = 0; i < N; i++){                         \
    sum ## fn += fn(rand_double(&rng) * scale);        \
  }                                                      \
  DEBUG_TIMER(fn);                                       \
  DEBUG("%s float sum %f", QUOTE(fn), sum ## fn);


#define TR_START -5.1
#define TR_STOP   5.1
#define TR_STEP   3e-6

#define DR_START -2.1
#define DR_STOP   2.1
#define DR_START2 -7.1
#define DR_STOP2   7.1
#define DR_STEP   1e-4

int
main(void){
  test_float_range(tanhf, lambert_76_tanhf, -9, 9, 1.3);
  rand_ctx rng;
  init_rand64(&rng, 1);
  TIME_RANGE(tanhf, TR_START, TR_STOP, TR_STEP);
  TIME_RANGE(fast_tanhf, TR_START, TR_STOP, TR_STEP);
  TIME_RANGE(lambert_76_tanhf, TR_START, TR_STOP, TR_STEP);
  TIME_RANGE(lambert_56_tanhf, TR_START, TR_STOP, TR_STEP);
  TIME_RANGE(best_rational_tanhf, TR_START, TR_STOP, TR_STEP);
  TIME_RANGE(pade_tanhf, TR_START, TR_STOP, TR_STEP);
  TIME_RANGE(pade2_tanhf, TR_START, TR_STOP, TR_STEP);
  TIME_RANGE(fast_exp_tanhf, TR_START, TR_STOP, TR_STEP);

  DEBUG("small range");
  DELTA_RANGE(tanhf, fast_tanhf, DR_START, DR_STOP, DR_STEP);
  DELTA_RANGE(tanhf, lambert_76_tanhf, DR_START, DR_STOP, DR_STEP);
  DELTA_RANGE(tanhf, lambert_56_tanhf, DR_START, DR_STOP, DR_STEP);
  DELTA_RANGE(tanhf, pade_tanhf, DR_START, DR_STOP, DR_STEP);
  DELTA_RANGE(tanhf, pade2_tanhf, DR_START, DR_STOP, DR_STEP);
  DELTA_RANGE(tanhf, best_rational_tanhf, DR_START, DR_STOP, DR_STEP);
  DELTA_RANGE(tanhf, fast_exp_tanhf, DR_START, DR_STOP, DR_STEP);
  
  DEBUG("big range");
  DELTA_RANGE(tanhf, fast_tanhf, DR_START2, DR_STOP2, DR_STEP);
  DELTA_RANGE(tanhf, lambert_76_tanhf, DR_START2, DR_STOP2, DR_STEP);
  DELTA_RANGE(tanhf, lambert_56_tanhf, DR_START2, DR_STOP2, DR_STEP);
  DELTA_RANGE(tanhf, pade_tanhf, DR_START2, DR_STOP2, DR_STEP);
  DELTA_RANGE(tanhf, pade2_tanhf, DR_START2, DR_STOP2, DR_STEP);
  DELTA_RANGE(tanhf, best_rational_tanhf, DR_START2, DR_STOP2, DR_STEP);
  DELTA_RANGE(tanhf, fast_exp_tanhf, DR_START2, DR_STOP2, DR_STEP);
}
