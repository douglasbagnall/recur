/* Copyright (C) 2013 Douglas Bagnall <douglas@halo.gen.nz> */
#ifndef _GOT_RECUR_NN_H
#define _GOT_RECUR_NN_H 1

#include <unistd.h>
#include <string.h>
#include "recur-common.h"
#include "recur-rng.h"

#include <cblas.h>
#include "pgm_dump.h"

#define VECTOR 1
#define VECTOR_ALL_THE_WAY (1 && VECTOR)

#if VECTOR
typedef float v4ss __attribute__ ((vector_size (16))) __attribute__ ((aligned (16)));
#endif

#define RECUR_RNG_SUBSEED -2ULL

/*controls magnitude of random damage (if that is used):
  variance = RANDOM_DAMAGE_FACTOR * net->h_size * net->bptt->learn_rate */
#define RANDOM_DAMAGE_FACTOR 0.5f

/*{MIN,MAX}_*_ERROR_FACTORs control how deep the bptt training goes.

  If the backpropagated numer drops below a value based on the MIN settings,
  or above a number based on MAX, the back-propagation loop stops.

  See bptt_and_accumulate_error()
*/
#define MAX_TOP_ERROR_FACTOR 2.0f
/*if bptt error grows by more than MAX_ERROR_GAIN, abort and scale */
#define MAX_ERROR_GAIN 2.0f
/*BASE_MIN_ERROR_FACTOR relates to initial minimum mean error*/
#define BASE_MIN_ERROR_FACTOR 1e-11f
/*min_error_factor never goes below ABS_MIN_ERROR_FACTOR*/
#define ABS_MIN_ERROR_FACTOR 1e-20f
/*MIN_ERROR_GAIN */
#define MIN_ERROR_GAIN 1e-8f
/* RNN_HIDDEN_PENALTY is subtracted from each hidden node, forcing low numbers to zero.
 1e-3f is safe, but less accurate than 1e-4f
XXX this really ought to be adjustable or adjust itself */
#define RNN_HIDDEN_PENALTY 1e-4f
/*scaling for hidden and input numbers */
#define HIDDEN_MEAN_SOFT_TOP 16.0f
#define INPUT_MEAN_SOFT_TOP 4.0f

#define RNN_INITIAL_WEIGHT_VARIANCE_FACTOR 10.0f
#define WEIGHT_SCALE (1.0f - 1e-6f)

/*RNN_CONDITIONING_INTERVAL should be <= 32, ideally a power of 2 */
#define RNN_CONDITIONING_INTERVAL 8

#define RNN_TALL_POPPY_THRESHOLD 1.0f
#define RNN_TALL_POPPY_SCALE 0.99f
#define RNN_LAWN_MOWER_THRESHOLD 10.0f

/* Conditioning flags go in bits 16-23 of net->flags.

   The RNN_COND_BIT_* numbers indicate the points in the conditioning cycle
   that various kinds of work will be done. They are deliberately sparsely
   spaced.

   The RNN_COND_USE_* flags turn on the respective conditioning task.
 */

#define RNN_COND_USE_OFFSET 16

enum {
  RNN_COND_BIT_SCALE = 0U,
  RNN_COND_BIT_ZERO = 2U,
  RNN_COND_BIT_LAWN_MOWER = 3U,
  RNN_COND_BIT_TALL_POPPY = 4U,
  RNN_COND_BIT_RAND = 6U
};

enum {
  RNN_NET_FLAG_OWN_BPTT = 1,
  RNN_NET_FLAG_OWN_WEIGHTS = 2,
  RNN_NET_FLAG_BIAS = 4,
  RNN_NET_FLAG_LOG_APPEND = 8,
  RNN_NET_FLAG_LOG_HIDDEN_SUM = 16, /*log the hidden sum */
  RNN_NET_FLAG_LOG_WEIGHT_SUM = 32, /*log the weight sum (can be expensive)*/
  RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR = 64, /*min error threshold auto-adjusts*/

  /*conditioning flags start at 1 << 16 (65536) */
  RNN_COND_USE_SCALE = (1 << (RNN_COND_BIT_SCALE + RNN_COND_USE_OFFSET)),
  RNN_COND_USE_ZERO = (1 << (RNN_COND_BIT_ZERO + RNN_COND_USE_OFFSET)),
  RNN_COND_USE_LAWN_MOWER = (1 << (RNN_COND_BIT_LAWN_MOWER + RNN_COND_USE_OFFSET)),
  RNN_COND_USE_TALL_POPPY = (1 << (RNN_COND_BIT_TALL_POPPY + RNN_COND_USE_OFFSET)),
  RNN_COND_USE_RAND = (1 << (RNN_COND_BIT_RAND + RNN_COND_USE_OFFSET)),

  /*more flags can fit after 1 << 24 or so */

  RNN_NET_FLAG_STANDARD = (RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS \
       | RNN_COND_USE_ZERO | RNN_NET_FLAG_BIAS | RNN_NET_FLAG_LOG_HIDDEN_SUM),
  RNN_NET_FLAG_NO_BIAS = RNN_NET_FLAG_STANDARD & ~ RNN_NET_FLAG_BIAS
};


typedef struct _RecurNN RecurNN;
typedef struct _RecurNNBPTT RecurNNBPTT;

struct _RecurNN {
  /*aligned sizes, for quick calculation */
  int i_size; /*includes hidden feedback and bias */
  int h_size;
  int o_size;
  /*actual requested sizes*/
  int input_size;
  int hidden_size;
  int output_size;
  /*matrix sizes */
  int ih_size;
  int ho_size;
  int bias;
  u32 flags;
  FILE *log;
  float *mem;
  float *input_layer;
  float *hidden_layer;
  float *output_layer;
  float *ih_weights;
  float *ho_weights;
  float *real_inputs;
  rand_ctx rng;
  RecurNNBPTT *bptt; /*training struct*/
  u32 generation;
};

struct _RecurNNBPTT {
  int depth;
  int index;
  float *i_error;
  float *h_error;
  float *o_error;
  float *ih_momentum;
  float *ho_momentum;
  float *history;
  float *ih_delta;
  float *ho_delta;
  float *mem;
  float learn_rate;
  float ih_scale;
  float ho_scale;
  float momentum;
  float momentum_weight;
  int batch_size;
  float min_error_factor;
};

#define ALIGNED_SIZEOF(x)  ((sizeof(x) + 15UL) & ~15UL)
#define ALIGNED_VECTOR_LEN(x, type) ((((x) * sizeof(type) + 15UL) & ~15UL) / sizeof(type))

/* functions */

RecurNN * rnn_new(uint input_size, uint hidden_size, uint output_size,
    int flags, u64 rng_seed, const char *log_file, int depth, float learn_rate,
    float momentum, float momentum_weight, int batch_size, int weight_shape);

RecurNN * rnn_clone(RecurNN *parent, int flags,
    u64 rng_seed, const char *log_file);

void rnn_set_log_file(RecurNN *net, const char * log_file, int append_dont_truncate);
void rnn_fd_dup_log(RecurNN *net, RecurNN* src);

void rnn_randomise_weights(RecurNN *net, float variance, int power);

void rnn_delete_net(RecurNN *net);

float *rnn_opinion(RecurNN *net, const float *inputs);
float *rnn_opinion_with_dropout(RecurNN *net, const float *inputs, float dropout);

void rnn_multi_pgm_dump(RecurNN *net, const char *dumpees);

RecurNN* rnn_load_net(const char *filename);
int rnn_save_net(RecurNN *net, const char *filename);

void bptt_advance(RecurNN *net);
void bptt_calculate(RecurNN *net);

void bptt_consolidate_many_nets(RecurNN **nets, int n, int nestorov,
    float momentum_soft_start);

void bptt_calc_deltas(RecurNN *net);

void rnn_condition_net(RecurNN *net);
void rnn_log_net(RecurNN *net);

void rnn_forget_history(RecurNN *net, int bptt_too);

static inline void
bptt_log_float(RecurNN *net, char *name, float value){
  if (net->log){
    fprintf(net->log, "%s %.5g\n", name, value);
  }
}

static inline void
bptt_log_int(RecurNN *net, char *name, int value){
  if (net->log){
    fprintf(net->log, "%s %d\n", name, value);
  }
}

static inline void
scale_aligned_array(float *array, int len, float scale)
{
  /*Using cblas might create denormal numbers -- unless the blas library has
    been compiled with -ffastmath. */
#if 1
  ASSUME_ALIGNED(array);
  for (int i = 0; i < len; i++){
    array[i] *= scale;
  }
#else
  cblas_sscal(len, scale, array, 1);
#endif
}

static inline void
add_aligned_arrays(float *restrict dest, int len, const float *restrict src, float scale)
{
  /*dest = dest + src * scale
    cblas_saxpy can do it. */
#if 1
  //XXX a prefetch would help
  ASSUME_ALIGNED(dest);
  ASSUME_ALIGNED(src);
#if VECTOR_ALL_THE_WAY && 0
  len >>= 2;
  v4ss *vd = (v4ss*)dest;
  v4ss *vs = (v4ss*)src;
  v4ss v_scale = {scale, scale, scale, scale};
  for (int i = 0; i < len; i++){
    __builtin_prefetch(&vs[i + 3]);
    __builtin_prefetch(&vd[i + 3]);
    vd[i] += vs[i] * v_scale;
  }

#else
  for (int i = 0; i < len; i++){
    dest[i] += src[i] * scale;
  }
#endif
#else
  cblas_saxpy(len, scale, src, 1, dest, 1);
#endif
}

static inline void
dropout_array(float *array, int len, float dropout, rand_ctx *rng){
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
  float x = sum / halfmax;
  float fudge = 0.99 + x * x / 100;
  return 2.0f * x / (1 + x * x * fudge);
  //((2 * x) / (1 + x * x)) / (0.99 + abs(x / 100))
  //float fudge = 0.99 + sum * sum / (halfmax * halfmax * 100);
  //return 2.0f * sum * halfmax / (halfmax * halfmax + sum * sum * fudge);
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

#endif
