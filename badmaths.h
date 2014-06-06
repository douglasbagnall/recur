/* Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL/MPL2 */
#ifndef HAVE_BADMATHS_H
#define HAVE_BADMATHS_H
/*XXX it would be possible to vectorise and do 4 at a time */

#define SYMMETRIC_EXTREMA_CLAMP(x, extreme, low, high) do {     \
  if ((x) <= -(extreme)) return (low);                          \
  if ((x) >= (extreme)) return (high); } while(0)


/*Pade 2, 2 approximant
Mineiro has a faster less accurate method.
*/
static inline float __attribute__ ((always_inline))
fast_expf(float x){
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

#define SIGMOID_SCALE 1.0f

static inline float
fast_sigmoid(float x){
  return 1.0f / (1.0f + fast_expf(-x * SIGMOID_SCALE));
}

static inline void
fast_sigmoid_array(float *dest, float *src, int len)
{
  for (int i = 0; i < len; i++){
    dest[i] = 1.0f / (1.0f + fast_expf(-src[i]  * SIGMOID_SCALE));
  }
}

static inline void
fast_sigmoid_byte_array(u8 *dest, const float *src, int len)
{
  for (int i = 0; i < len; i++){
    dest[i] = 255.99f / (1.0f + fast_expf(-src[i] * SIGMOID_SCALE));
  }
}


static inline float
fast_tanhf(float x)
{
  /*based on
    http://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
  */
  /*rather than simply clamp, it would be possible to scale and use
   identities. but the answer is close to +/- 1 anyway.*/
  SYMMETRIC_EXTREMA_CLAMP(x, 4.97f, -1.0f, 1.0f);
  float x2 = x * x;
  float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
  float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
  return a / b;
}


static inline void
softmax(float *restrict dest, const float *restrict src, int len){
  int i;
  float sum = 0.0f;
  float adj = 0.0f;
  float min = src[0];
  float max = src[0];
  const float max_exp = 50.0f;
  const float min_exp = -60.0f;
  /* for safety, avoid really big numbers. Addition/subtraction of a constant
     to the src vector doesn't affect the outcome (difference here becomes
     ratio post-exp). */
  for (i = 1; i < len; i++){
    max = MAX(max, src[i]);
    min = MIN(min, src[i]);
  }
  if (max > max_exp){
    adj = max_exp - max;
  }
  else if (min < min_exp){
    adj = MIN(min_exp - min, max_exp - max);
  }
  /*shifting most numbers toward zero would actually speed things up, because
    fast_expf would have less resizing to do in its loops -- but it is not
    clear that the midpoint of max and min will shift most numbers toward
    zero. Finding the mean might work better, though it is easy to imagine
    pathological cases there.
   */
  //else {
  //  adj = -0.5 * (max + min);
  //}
  for (i = 0; i < len; i++){
    float f = src[i] + adj;
    float x = fast_expf(f);
    sum += x;
    dest[i] = x;
  }
  for (i = 0; i < len; i++){
    dest[i] /= sum;
  }
}

static inline int
softmax_best_guess(float *restrict error, const float *restrict src, int len)
{
  softmax(error, src, len);
  /*softmax error is 0-1. all values should be 0, EXCEPT the hot one, which
   should be 1. The passed in error array is overwritten with negated softmax
   values. Training error encodes the desired change, i.e., a negative number
   in most cases, and 1 - softmax for the correct answer, so in training
   situations you want to add one straight afterwards:

    int target = whatever();
    int winner = softmax_best_guess(error, answer, len);
    error[target] += 1.0f;

   Sum of softmax is always one, so error sum is always twice target error.
  */
  int best_i = 0;
  float best_e = error[0];
  error[0] = -best_e;
  for (int i = 1; i < len; i++){
    float e = error[i];
    if (e > best_e){
      best_e = e;
      best_i = i;
    }
    error[i] = -e;
  }
  return best_i;
}

static inline void
biased_softmax(float *restrict dest, const float *restrict src, int len, float bias){
  if (bias == 0){
    softmax(dest, src, len);
  }
  else {
    float tmp[len];
    softmax(tmp, src, len);
    for (int i = 0; i < len; i++){
      tmp[i] = tmp[i] * bias + src[i];
    }
    softmax(dest, tmp, len);
  }
}

#endif
