/* Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL/MPL2 */
#ifndef _GOT_RECUR_RNG_H
#define _GOT_RECUR_RNG_H 1
#include "recur-common.h"
#include <time.h>
#include <math.h>
/** Random numbers **/

/*Based on the 64 bit version on Bob Jenkins' small fast pseudorandom number
  generator at http://burtleburtle.net/bob/rand/smallprng.html
  Public domain.
*/
#define RECUR_RNG_RANDOM_SEED -1ULL

typedef struct _rand_ctx {
  u64 a;
  u64 b;
  u64 c;
  u64 d;
} rand_ctx;

#define ROTATE(x, k) (((x) << (k)) | ((x) >> (sizeof(x) * 8 - (k))))

static inline u64
rand64(rand_ctx *x)
{
  u64 e = x->a - ROTATE(x->b, 7);
  x->a = x->b ^ ROTATE(x->c, 13);
  x->b = x->c + ROTATE(x->d, 37);
  x->c = x->d + e;
  x->d = e + x->a;
  return x->d;
}


static inline void
init_rand64(rand_ctx *x, u64 seed)
{
  int i;
  x->a = 0xf1ea5eed;
  x->b = x->c = x->d = seed;
  for (i = 0; i < 20; ++i) {
    (void)rand64(x);
  }
}

static inline void
init_rand64_maybe_randomly(rand_ctx *ctx, u64 seed)
{
  if (seed == RECUR_RNG_RANDOM_SEED){
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    seed = (u64)(t.tv_nsec + t.tv_sec) ^ (u64)ctx;
    DEBUG("seeding with %zx\n", seed);
  }
  init_rand64(ctx, seed);
}

#define DSFMT_LOW_MASK  0x000FFFFFFFFFFFFFUL
#define DSFMT_HIGH_CONST 0x3FF0000000000000UL
#define DSFMT_SR	12
#define DSFMT_INT64_TO_DOUBLE(x) (((x) & DSFMT_LOW_MASK) | DSFMT_HIGH_CONST)

#define MT_U64_TO_2_F32_LOW_MASK 0x7fffff007fffffUL
#define MT_U64_TO_2_F32_HIGH_CONST 0x3F8000003F800000UL
#define MT_U32_TO_F32_LOW_MASK 0x007fffff
#define MT_U32_TO_F32_HIGH_CONST 0x3F800000
#define MT_U32_TOF32_SHIFT_RIGHT 9


static inline double
rand_double(rand_ctx *rng)
{
  union{
    u64 i;
    double d;
  } x;
  x.i = DSFMT_INT64_TO_DOUBLE(rand64(rng));
  return x.d - 1.0;
}

static inline double
rand_expovariate(rand_ctx *rng, double lambda)
{
  double d = rand_double(rng);
  return - log(1.0 - d) / lambda;
}


/*this method has a bias that is noticeable with large integers. */
static inline int
rand_small_int(rand_ctx *rng, int cap){
  double d = rand_double(rng) * cap;
  return (int)d;
}

#define RAND_SMALL_INT_RANGE(rng, start, cap) \
  ((start) + rand_small_int(rng, (cap) - (start)))

/*recipricals of 1<<31, 1<<62 */
#define RECIP31f 4.656612873077393e-10f
#define RECIP62f 2.168404344971009e-19f

/*Generate two standard normal numbers using what Wikipedia calls the
  "Marsaglia polar method", using fixed point arithmetic. It turns out this is
  slightly slower than using floating point, though it is possibly more
  precise (because all 64 random bits are used).
*/
static inline void
doublecheap_gaussian_noise_f(rand_ctx *ctx,
    float *restrict a, float *restrict b,
    const float deviation){
  u64 r;
  s64 x, y;
  union {
    s32 s[2];
    u64 u;
  } i;
  for(;;){
    /*calculate in fixed point: x, y are 32 bit signed; r is 62 bit*/
    i.u = rand64(ctx);
    x = i.s[0];
    y = i.s[1];
    r = x * x + y * y;
    if (r && r < (1UL << 62UL))
      break;
  }
  float s = (float)r * RECIP62f;
  float m = sqrtf(-2.0f * logf(s) / s) * RECIP31f * deviation;
  *a = x * m;
  *b = y * m;
}

/*Generate two standard normal numbers using what Wikipedia calls the
  "Marsaglia polar method", using floating point arithmetic.
*/
static inline void
doublecheap2_gaussian_noise_f(rand_ctx *ctx, float *restrict a,
    float *restrict b, const float deviation){
  float s;
  union {
    float s[2];
    u64 i;
  } x;
  float c, d;
  for(;;){
    u64 i = rand64(ctx);
    x.i = i & MT_U64_TO_2_F32_LOW_MASK;
    x.i |= MT_U64_TO_2_F32_HIGH_CONST;

    c = x.s[0] + x.s[0] - 3.0f;
    d = x.s[1] + x.s[1] - 3.0f;
    s = c * c + d * d;

    if (s < 1.0f && s != 0.0f)
      break;
  }
  float m = sqrtf(-2.0f * logf(s) / s) * deviation;
  *a = c * m;
  *b = d * m;
}


/*cheap_gaussian_noise is actually an Irwin-Hall distribution (i.e. a
  central-limit-theorem based discrete approximation of the standard normal).

  It has low precision -- roughly 19 bits -- and hard limits at +/- 6 standard
  deviations. In practice this is barely worse than the 32 bit float methods
  (which are theoretically contrained at 6.6 bits), while being substantially
  quicker and easier to use because it generates and returns one sample at a
  time.
*/

static inline float
cheap_gaussian_noise(rand_ctx *ctx){
  s64 a = 0;
  u64 i = rand64(ctx);
#define _add_16_bits() a += i & 0xffff; i >>= 16;
  _add_16_bits();
  _add_16_bits();
  _add_16_bits();
  _add_16_bits();
  i = rand64(ctx);
  _add_16_bits();
  _add_16_bits();
  _add_16_bits();
  _add_16_bits();
  i = rand64(ctx);
  _add_16_bits();
  _add_16_bits();
  _add_16_bits();
  _add_16_bits();
#undef _add_16_bits
return (float)(a - 0xffff * 6) / (0xffff);
}


/*randomise_mem sets <size> bytes to random bits */
static inline void
randomise_mem(rand_ctx *rng, void *mem, const size_t size)
{
  size_t i;
  u64 *m64 = mem;
  size_t size64 = size / sizeof(u64);
  for (i = 0; i < size64; i++){
    m64[i] = rand64(rng);
  }
  if (size > size64 * sizeof(u64)){
    u64 r = rand64(rng);
    u8 *bytes = (u8 *)r;
    for (i = size; i < size64 * sizeof(u64); i++){
      m64[i] = bytes[i];
    }
  }
}



/*randomise_float_array sets <len> floats to (0-1] */
static inline void
randomise_float_array(rand_ctx *rng, float *array, int size)
{
  int i;
  if (size & 1){
    size -= 1;
    u32 *a = (void*)(&array[size]);
    *a = (u32)rand64(rng);
    *a &= MT_U32_TO_F32_LOW_MASK;
    *a |= MT_U32_TO_F32_HIGH_CONST;
  }
  for (i = 0; i < size; i += 2){
    u64 *a = (void*)(&array[i]);
    *a = rand64(rng);
    *a &= MT_U64_TO_2_F32_LOW_MASK;
    *a |= MT_U64_TO_2_F32_HIGH_CONST;
  }
  for (i = 0; i < size; i ++){
    array[i] -= 1.0f;
  }
}

static inline void
randomise_float_pair(rand_ctx *rng, float *f){
  u64 *a = (void*)f;
  *a = rand64(rng);
  *a &= MT_U64_TO_2_F32_LOW_MASK;
  *a |= MT_U64_TO_2_F32_HIGH_CONST;
  f[0] -= 1.0f;
  f[1] -= 1.0f;
}


#endif
