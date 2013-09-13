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
  float sum = 0.0;
  for (i = 0; i < len; i++){
    float f = src[i];
    f = MIN(f, 60);
    f = MAX(f, -60);
    float x = fast_expf(f);
    sum += x;
    dest[i] = x;
  }
  for (i = 0; i < len; i++){
    dest[i] /= sum;
  }
}

static inline float
softmax_best_guess(float *restrict error, const float *restrict src,
                   int len, int target, int *winner)
{
  softmax(error, src, len);
  /*softmax error is 0-1. all values should be 0, EXCEPT the hot one, which
   should be 1. Error encodes the desired change, i.e., a negative number in
   most cases, and 1 - softmax for the hot.

   Sum of softmax is always one, so error sum is always twice target error.
  */
  int best_i = -1;
  float best_e = -1;
  for (int i = 0; i < len; i++){
    if (error[i] > best_e){
      best_e = error[i];
      best_i = i;
    }
    error[i] = -error[i];
  }
  //DEBUG("best guess %d next %d", best_i, next);
  if (winner){
    *winner = best_i;
  }
  error[target] += 1.0f;
  return error[target];
}


#endif
