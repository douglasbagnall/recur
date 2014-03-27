#include "recur-nn.h"
#include "pgm_dump.h"
#include <math.h>

#define F32_EXPONENT      0x3C800000
#define F32_EXP_ONE       0x00800000
#define F32_MANTISSA_MASK 0x807FFFFF
#define F32_EXP_MASK      0x7F800000

#define F64_EXPONENT       0x3F90000000000000UL
#define F64_EXP_ONE        0x0010000000000000UL
#define F64_MANTISSA_MASK  0x800FFFFFFFFFFFFFUL
#define F64_EXP_MASK       0x7FF0000000000000UL

#define E 2.718281828459045
#define Ef 2.718281828459045f
#define LG2 0.6931471805599453
#define LG2f 0.6931471805599453f

#define THIRD     0.33333333333333333333
#define FIFTH     0.2
#define SEVENTH   0.14285714285714285714
#define NINTH     0.11111111111111111111
#define ELEVENTH  0.09090909090909090909

#define THIRDf     0.33333333333333333333f
#define FIFTHf     0.2f
#define SEVENTHf   0.14285714285714285714f
#define NINTHf     0.11111111111111111111f
#define ELEVENTHf  0.09090909090909090909f

/* from cephes http://www.netlib.org/cephes/doubldoc.html
 *     log(1+x) = x - 0.5 x**2 + x**3 P(x)/Q(x).
 *
 * Otherwise, setting  z = 2(x-1)/x+1),
 *
 *     log(x) = z + z**3 P(z)/Q(z)


 *
 * A Pade' form  1 + 2x P(x**2)/( Q(x**2) - P(x**2) )
 * of degree 2/3 is used to approximate exp(f) in the basic
 * interval [-0.5, 0.5].
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    DEC       +- 88       50000       2.8e-17     7.0e-18
 *    IEEE      +- 708      40000       2.0e-16     5.6e-17
 *
 *
 */


/*untried newcomers */
/*https://www.allegro.cc/forums/thread/183952/183964 #1*/
static float fast_logf_allegro(float x)
{
  float y = (x - 1.0f) / (x + 1.0f);
  float y_squared = y * y;
  return (y * 2.0f * (15.0f - y_squared * 4.0f) / (15.0f - y_squared * 9.0f));
}

/****https://www.allegro.cc/forums/thread/183952/183964 #2 ****/
static inline float fast_logf_small(float x)
{
  float y = (x-1) / (x+1);
  float y_cubed = y * y * y;
  return 2.0f * (y + 0.33333333f * y_cubed + 0.2f * y_cubed * y * y);
}

#define MANTISSA(x)((x) & 0x7FFFFF) * 1.1920928955e-7 + 1.0
#define EXPONENT(x) ((((x) >> 23) & 255) - 127)

static float
fast_logf_43(float y)
{
  union REAL {
    u32 i;
    float fp;
  } x;
  x.fp = y;
  float m = MANTISSA(x.i);
  int exp = EXPONENT(x.i);
  return (fast_logf_small(m) + (float)exp) * LG2f;
}

/******************/

static float
fast_logf(float x){
  float c = 0, a;
  for (; x > 2.0f; x /= Ef, c++){}
  for (; x < 0.5f; x *= Ef, c--){}
  float y = (x - 1.0f) / (x + 1.0f);
  float y2 = y * y;
  a = 2.0f * y * (1.0f + y2 * (THIRDf + y2 * (FIFTHf + y2 * (SEVENTHf + y2 * NINTHf))));
  return a + c;
}

static float
fast_logf_b(float x){
  union {
    u32 i;
    float f;
  } b;
  float a;
  b.f = x;
  u32 exponent = b.i & F32_EXP_MASK;
  b.i &= F32_MANTISSA_MASK;
  b.i |= 0x3F000000; /*0.5 to 1.0 */
  x = b.f;
  float c = (int)((exponent >> 23) - 126) * LG2f;
  float y = (x - 1.0f) / (x + 1.0f);
  float y2 = y * y;
  a = 2.0f * y * (1.0f + y2 * (THIRDf + y2 * (FIFTHf + y2 * (SEVENTHf + y2 * NINTHf))));
  return a + c;
}

static float
fast_logf_b_pade(float x){
  union {
    u32 i;
    float f;
  } b;
  b.f = x;
  u32 exponent = b.i & F32_EXP_MASK;
  b.i &= F32_MANTISSA_MASK;
  b.i |= 0x3F000000; /*0.5 to 1.0 */
  x = b.f;
  float c = (int)((exponent >> 23) - 126) * LG2f;
  float y = x - 1.0f;
  float a = y * (6.0f + y) / (6.0f + 4.0f * y);
  return a + c;
}


static float
fast_logf_mineiro(float x){
  union {
    float f;
    u32 i;
  } vx = { x };
  union {
    u32 i;
    float f;
  } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
  float y = vx.i;
  y *= 1.1920928955078125e-7f;

  return (y - 124.22551499f
            - 1.498030302f * mx.f
      - 1.72587999f / (0.3520887068f + mx.f)) * LG2f;
}


static float
fast_logf_mineiro2 (float x)
{
  union { float f; uint32_t i; } vx = { x };
  float y = vx.i;
  y *= 8.2629582881927490e-8f;
  return y - 87.989971088f;
}


static double
fast_log(double x){
  double c, a;
  for (c = 0; x > 2.0; x /= E, c++){}
  for (; x < 0.5; x *= E, c--){}
  double y = (x - 1.0) / (x + 1.0);
  double y2 = y * y;
  a = 2 * y * (1.0 + y2 * (THIRD + y2 * (FIFTH + y2 * (SEVENTH + y2 * (NINTH + y2 * (ELEVENTH))))));
  return a + c;
}



static double
fast_log_b_pade(double x){
  union {
    u64 i;
    double d;
  } b;
  b.d = x;
  u64 exponent = b.i & F64_EXP_MASK;
  b.i &= F64_MANTISSA_MASK;
  b.i |= 0x3FE0000000000000UL; /*0.5 to 1.0 */
  x = b.d;
  double c = (int)((exponent >> 52) - 0x3fE) * LG2;
  double y = (x - 1.0);
  double a = y * (6 + y) / (6 + 4 * y);
  //DEBUG("x %f a %f c %f exponent %lx", x, a, c, (exponent >> 52) -0x3fe);
  return a + c;
}


static double
fast_log_b(double x){
  union {
    u64 i;
    double d;
  } b;
  b.d = x;
  u64 exponent = b.i & F64_EXP_MASK;
  b.i &= F64_MANTISSA_MASK;
  b.i |= 0x3FE0000000000000UL; /*0.5 to 1.0 */
  x = b.d;
  double c = (int)((exponent >> 52) - 0x3fe) * LG2;
  double y = (x - 1.0) / (x + 1.0);
  double y2 = y * y;
  double a = 2 * y * (1.0 + y2 * (THIRD + y2 * (FIFTH + y2 * (SEVENTH + y2 * (NINTH + y2 * (ELEVENTH))))));
  //DEBUG("x %f a %f c %f exponent %lx", x, a, c, (exponent >> 52) -0x3fe);
  return a + c;
}


static float
fastpow2_mineiro (float p)
{
  float offset = (p < 0) ? 1.0f : 0.0f;
  float clipp = (p < -126) ? -126.0f : p;
  int w = clipp;
  float z = clipp - w + offset;
  union { uint32_t i; float f; } v = { (u32) ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };
  return v.f;
}

static float
fast_expf_mineiro (float p)
{
  return fastpow2_mineiro (1.442695040f * p);
}



static float
fast_expf_taylor3(float x){
  union {
    u32 i;
    float f;
  } b;
  float a;
  b.f = x;
  u32 exponent = b.i & F32_EXP_MASK;
  if (exponent > F32_EXPONENT){
    b.i &= F32_MANTISSA_MASK;
    b.i |= F32_EXPONENT;
    x = b.f;
  }
  a = 1 + x + x * x / 2 + x * x * x / 6;
  while(exponent > F32_EXPONENT){
    a *= a;
    exponent -= F32_EXP_ONE;
  }
  return a;
}


static float
fast_expf_taylor2(float x){
  union {
    u32 i;
    float f;
  } b;
  float a;
  b.f = x;
  u32 exponent = b.i & F32_EXP_MASK;
  if (exponent > F32_EXPONENT){
    b.i &= F32_MANTISSA_MASK;
    b.i |= F32_EXPONENT;
    x = b.f;
  }
  a = 1 + x + x * x / 2 + x * x * x / 6 + x * x * x * x / 24;
  while(exponent > F32_EXPONENT){
    a *= a;
    exponent -= F32_EXP_ONE;
  }
  return a;
}



static float
fast_expf_taylor(float x){
  int count = 0;
  while (fabsf(x) > 0.2){
    x *= 0.125;
    count++;
  }
  float a = 1 + x + x * x / 2 + x * x * x / 6 + x * x * x * x / 24;
  while(count){
    a *= a;
    a *= a;
    a *= a;
    count--;
  }
  return a;
}

static float
fast_expf_22(float x){
  int count = 0;
  while (fabsf(x) > 0.2){
    x *= 0.125;
    count++;
  }
  float a = ((x + 3) * (x + 3) + 3) / ((x - 3) * (x - 3) + 3);
  while(count){
    a *= a;
    a *= a;
    a *= a;
    count--;
  }
  return a;
}





static double
fast_exp_taylor2(double x){
  union {
    u64 i;
    double d;
  } b;
  double a;
  b.d = x;
  u64 exponent = b.i & F64_EXP_MASK;
  if (exponent > F64_EXPONENT){
    b.i &= F64_MANTISSA_MASK;
    b.i |= F64_EXPONENT;
    x = b.d;
  }
  a = 1 + x + x * x / 2 + x * x * x / 6 + x * x * x * x / 24;
  while(exponent > F64_EXPONENT){
    a *= a;
    exponent -= F64_EXP_ONE;
  }
  return a;
}

static double
fast_exp_taylor(double x){
  int count = 0;
  while (fabs(x) > 0.1){
    x *= 0.125;
    count++;
  }
  double a = 1 + x + x * x / 2 + x * x * x / 6 + x * x * x * x / 24;
  while(count){
    a *= a;
    a *= a;
    a *= a;
    count--;
  }
  return a;
}

static double
fast_exp_22(double x){
  int count = 0;
  while (fabs(x) > 0.1){
    x *= 0.125;
    count++;
  }
  double a = ((x + 3) * (x + 3) + 3) / ((x - 3) * (x - 3) + 3);
  /*double x2 = x * x + 12;
  double sixx = 6 * x;
  double a = (x2 + sixx) / (x2 - sixx);*/
  while(count){
    a *= a;
    a *= a;
    a *= a;
    count--;
  }
  return a;
}

static double
fast_exp_33(double x){
  int count = 0;
  while (fabs(x) > 0.1){
    x *= 0.125;
    count++;
  }
  /*
    1 + 1/2z + 1/10zz + 1/120zzz
    ----------------------------
    1 - 1/2z + 1/10zz - 1/120zzz

    * 120

    120 + 60z + 12zz + zzz
    ----------------------
    120 - 60z + 12zz - zzz

    (120 + 12zz) + (zzz + 60z)
    ----------------------
    (120 + 12zz) - (zzz + 60z)
   */
  double x2 = 12 * (10 + x * x);
  double x3 = x * (x * x + 60);
  double a = (x2 + x3) / (x2 - x3);
  while(count){
    a *= a;
    a *= a;
    a *= a;
    count--;
  }
  return a;
}


static double
fast_exp_bits_33(double x){
  union {
    u64 i;
    double d;
  } b;
  b.d = x;
  u64 exponent = b.i & F64_EXP_MASK;
  if (exponent > F64_EXPONENT){
    b.i &= F64_MANTISSA_MASK;
    b.i |= F64_EXPONENT;
    x = b.d;
  }
  double x2 = 12 * (10 + x * x);
  double x3 = x * (x * x + 60);
  double a = (x2 + x3) / (x2 - x3);
  while(exponent > F64_EXPONENT){
    a *= a;
    exponent -= F64_EXP_ONE;
  }
  return a;
}



static void UNUSED
test_float_range(float (*fn1)(float), float (*fn2)(float),
    float start, float stop, float step){
  float delta = 0.0;
  float count = 0.0;
  for (float f = start; f < stop; f += step){
    float a1 = fn1(f);
    float a2 = fn2(f);
    float d = (a1 - a2) / (a1 ? a1 : 1);
    DEBUG("%5.3f fn1 %f fn2 %f diff %g", f, a1, a2, d);
    delta += fabs(d);
    count ++;
  }
  DEBUG("average delta %g", delta / count);
}

static  UNUSED void
test_double_range(double (*fn1)(double), double (*fn2)(double),
    double start, double stop, double step){
  double delta = 0.0;
  double count = 0.0;
  for (double f = start; f < stop; f += step){
    double a1 = fn1(f);
    double a2 = fn2(f);
    double d = (a1 - a2) / (a1 ? a1 : 1);
    DEBUG("%5.3f fn1 %f fn2 %f diff %g", f, a1, a2, d);
    delta += fabs(d);
    count ++;
  }
  DEBUG("average delta %g", delta / count);
}




#define TIME_LOGF(fn, name)  \
  init_rand64(&rng, 1);     \
  START_TIMER(name); \
  float sum ## name = 0.0;                    \
  for (int i = 0; i < N; i++){              \
    sum ## name += fn(i + 0.1);     \
  }                                           \
  DEBUG_TIMER(name);                            \
  DEBUG("%s float sum %f", QUOTE(name), sum ## name);


#define TIME_LOG(fn, name)  \
  init_rand64(&rng, 1);     \
  START_TIMER(name); \
  double sum ## name = 0.0;                    \
  for (int i = 0; i < N; i++){              \
    sum ## name += fn(i + 0.1);     \
  }                                           \
  DEBUG_TIMER(name);                            \
  DEBUG("%s double sum %f", QUOTE(name), sum ## name);



#define TIME_EXPF(fn, name)  \
  init_rand64(&rng, 1);     \
  START_TIMER(name); \
  float sum ## name = 0.0;                    \
  for (int i = 0; i < N; i++){              \
    sum ## name += fn(rand_double(&rng));     \
  }                                           \
  DEBUG_TIMER(name);                            \
  DEBUG("%s float sum %f", QUOTE(name), sum ## name);


#define TIME_EXP(fn, name)  \
  init_rand64(&rng, 1);     \
  START_TIMER(name); \
  double sum ## name = 0.0;                    \
  for (int i = 0; i < N; i++){              \
    sum ## name += fn(rand_double(&rng));     \
  }                                           \
  DEBUG_TIMER(name);                            \
  DEBUG("%s double sum %f", QUOTE(name), sum ## name);



int
main(void){
  test_float_range(expf, fast_expf_mineiro, -10, 10, 1.4);
  //test_float_range(logf, fast_logf_mineiro2, 0.001, 170, 1.4);
  //test_double_range(exp, fast_exp_bits_33, -10, 10, 0.3);
  //test_double_range(log, fast_log_b_pade, 0.0001, 11, 0.4);
  rand_ctx rng;
  init_rand64(&rng, 1);
  int N = 10000000;

  TIME_EXPF(fast_expf_mineiro, f_mineiro);
  TIME_EXPF(fast_expf_taylor, f_taylor);
  TIME_EXPF(fast_expf_taylor2, f_taylor2);
  TIME_EXPF(fast_expf_taylor3, f_taylor3);
  TIME_EXPF(fast_expf_22, f_22);
  TIME_EXPF(expf, libm_expf);
  TIME_EXP(exp, libm_exp);
  TIME_EXP(fast_exp_22, double_22);
  TIME_EXP(fast_exp_33, double_33);
  TIME_EXP(fast_exp_bits_33, double_bits_33);
  TIME_EXP(fast_exp_taylor, double_taylor);
  TIME_EXP(fast_exp_taylor2, double_taylor2);

  DEBUG("*********** log ***********");
  TIME_LOGF(fast_logf, fast_logf);
  TIME_LOGF(fast_logf_b, fast_logf_b);
  TIME_LOGF(fast_logf_b_pade, fast_logf_b_pade);
  TIME_LOGF(fast_logf_mineiro2, fast_logf_mineiro2);
  TIME_LOGF(fast_logf_mineiro, fast_logf_mineiro);
  TIME_LOGF(fast_logf_allegro, fast_logf_allegro);
  TIME_LOGF(fast_logf_43, fast_logf_43);
  TIME_LOGF(logf, libm_logf);
  TIME_LOG(fast_log, fast_log);
  TIME_LOG(fast_log_b, fast_log_b);
  TIME_LOG(fast_log_b_pade, fast_log_b_pade);
  TIME_LOG(log, libm_log);
}
