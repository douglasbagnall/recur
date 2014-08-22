/* Copyright (C) 2013 Douglas Bagnall <douglas@halo.gen.nz>  LGPL

   These are mostly-private variables and helpers for recur-nn.
*/
#ifndef _GOT_RECUR_NN_HELPERS_H
#define _GOT_RECUR_NN_HELPERS_H 1

#include "recur-nn.h"

#if USE_CBLAS
#include <cblas.h>
#endif
#include "pgm_dump.h"

#if VECTOR
typedef float v4ss __attribute__ ((vector_size (16))) __attribute__ ((aligned (16)));
#endif

#define ALIGNED_SIZEOF(x)  ((sizeof(x) + 15UL) & ~15UL)
#define ALIGNED_VECTOR_LEN(x, type) ((((x) * sizeof(type) + 15UL) & ~15UL) / sizeof(type))

static inline void
scale_aligned_array(float *array, int len, float scale)
{
  /*Using cblas might create denormal numbers -- unless the blas library has
    been compiled with -ffastmath. */
#if ! USE_CBLAS
  ASSUME_ALIGNED(array);
  for (int i = 0; i < len; i++){
    array[i] *= scale;
  }
#else
  cblas_sscal(len, scale, array, 1);
#endif
}

static inline void
add_aligned_arrays(float *restrict dest, int len, const float *restrict src,
    const float scale)
{
  /*dest = dest + src * scale
    cblas_saxpy can do it. */
#if ! USE_CBLAS
  //XXX a prefetch would help
  ASSUME_ALIGNED(dest);
  ASSUME_ALIGNED(src);
#if VECTOR_ALL_THE_WAY && 0
  len >>= 2;
  v4ss *vd = (v4ss*)dest;
  v4ss *vs = (v4ss*)src;
  if (scale != 1.0f){
    v4ss v_scale = {scale, scale, scale, scale};
    for (int i = 0; i < len; i++){
      __builtin_prefetch(&vs[i + 3]);
      __builtin_prefetch(&vd[i + 3]);
      vd[i] += vs[i] * v_scale;
    }
  }
  else {
    for (int i = 0; i < len; i++){
      __builtin_prefetch(&vs[i + 3]);
      __builtin_prefetch(&vd[i + 3]);
      vd[i] += vs[i];
    }
  }

#else /*not VECTOR_ALL_THE_WAY*/
  if (scale == 1.0f){
    for (int i = 0; i < len; i++){
      dest[i] += src[i];
    }
  }
  else {
    for (int i = 0; i < len; i++){
      dest[i] += src[i] * scale;
    }
  }
#endif
#else
  cblas_saxpy(len, scale, src, 1, dest, 1);
#endif
}

static inline void
perforate_array(float *array, int len, float dropout, rand_ctx *rng){
  int i;
  if (dropout == 0.5f){ /*special case using far fewer random numbers*/
    for (i = 0; i < len;){
      u64 bits = rand64(rng);
      int end = i + MIN(64, len - i);
      for (; i < end; i++){
        array[i] = (bits & 1) ? array[i] : 0;
      }
    }
  }
  else {
    for (i = 0; i < len; i++){
      /*XXX could use randomise_float_pair() for possibly marginal speedup*/
      array[i] = (rand_double(rng) > dropout) ? array[i] : 0.0f;
    }
  }
}

static inline float
soft_clip(float sum, float halfmax){
  if (halfmax == 0){
    /*there is nothing sensible to do*/
    return sum;
  }
  float x = sum / halfmax;
  float fudge = 0.99 + x * x / 100;
  return 2.0f * x / (1 + x * x * fudge);
}

static inline float
softclip_scale(float sum, float halfmax, float *array, int len){
  ASSUME_ALIGNED(array);
  if (sum > halfmax){
    float scale = soft_clip(sum, halfmax);
    scale_aligned_array(array, len, scale);
    return scale * sum;
  }
  return sum;
}

static inline void
zero_small_numbers(float *array, int len)
{
  ASSUME_ALIGNED(array);
  for (int i = 0; i < len; i++){
    array[i] = (fabsf(array[i]) > 1e-34f) ? array[i] : 0.0f;
  }
}

static inline float
abs_sum_aligned_array(const float *array, int len)
{
  float sum = 0.0f;
  ASSUME_ALIGNED(array);
  for (int i = 0; i < len; i++){
    sum += fabsf(array[i]);
  }
  return sum;
}

#define ZERO_WITH_MEMSET 0

static inline void
zero_aligned_array(float *array, int size){
  ASSUME_ALIGNED(array);
#if ZERO_WITH_MEMSET
  memset(array, 0, size * sizeof(float));
#else
  int i;
#if VECTOR
  size >>= 2;
  v4ss *v_array = (v4ss*)array;
  v4ss vz = {0, 0, 0, 0};
  for (i = 0; i < size; i++){
    v_array[i] = vz;
  }
#else
  for (i = 0; i < size; i++){
    array[i] = 0.0f;
  }
#endif
#endif
}

static inline void
add_array_noise(rand_ctx *rng, float *array, int len, float deviation){
  for (int i = 0; i < len; i++){
    float noise = cheap_gaussian_noise(rng) * deviation;
    array[i] += noise;
  }
}

#define MAYBE_ADD_ARRAY_NOISE(rng, array, len, dev) do {        \
    if (dev)                                                    \
      add_array_noise(rng, array, len, dev);                    \
  } while(0)


#endif
