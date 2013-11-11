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

#define RANDOM_DAMAGE_MAGNITUDE 0.03

#define MAX_TOP_ERROR_FACTOR 2.0f
/*if bptt error grows by more than MAX_ERROR_GAIN, abort and scale */
#define MAX_ERROR_GAIN 2.0f
/*MIN_ERROR_FACTOR is minimum mean error */
#define MIN_ERROR_FACTOR 3e-8f
/* RNN_HIDDEN_PENALTY is subtracted from each hidden node, forcing low numbers to zero.
 1e-3f is safe, but less accurate than 1e-4f */
#define RNN_HIDDEN_PENALTY 1e-4f
/*scaling for hidden and input numbers */
#define HIDDEN_MEAN_SOFT_TOP 16.0f
#define INPUT_MEAN_SOFT_TOP 4.0f

#define RNN_INITIAL_WEIGHT_VARIANCE_FACTOR 10.0f
#define WEIGHT_SCALE (1.0f - 1e-6f)

/*RNN_CONDITIONING_INTERVAL should be <= 32, ideally a power of 2 */
#define RNN_CONDITIONING_INTERVAL 8

/*whether to log the hidden sum */
#define RNN_LOG_HIDDEN_SUM 1
/*whether to log the weight sum (relatively costly, often uninformative) */
#define RNN_LOG_WEIGHT_SUM 0

#define ASM_MARKER(x) asm("/**" QUOTE(x) "**/")

/* Conditioning flags go in bits 16-23 of net->flags.

   The RNN_COND_BIT_* numbers indicate the points in the conditioning cycle
   that various kinds of work will be done. They are deliberately sparsely
   spaced.

   The RNN_COND_USE_* flags turn on the respective conditioning task.
 */

#define RNN_COND_USE_OFFSET 16

enum {
  RNN_COND_BIT_SCALE = 0U,
  RNN_COND_BIT_ZERO = 3U,
  RNN_COND_BIT_RAND = 6U
};

enum {
  RNN_NET_FLAG_OWN_BPTT = 1,
  RNN_NET_FLAG_OWN_WEIGHTS = 2,
  RNN_NET_FLAG_BIAS = 4,
  RNN_NET_FLAG_LOG_APPEND = 8,

  /*conditioning flags start at 1 << 16 (65536) */
  RNN_COND_USE_SCALE = (1 << (RNN_COND_BIT_SCALE + RNN_COND_USE_OFFSET)),
  RNN_COND_USE_ZERO = (1 << (RNN_COND_BIT_ZERO + RNN_COND_USE_OFFSET)),
  RNN_COND_USE_RAND = (1 << (RNN_COND_BIT_RAND + RNN_COND_USE_OFFSET)),

  /*more flags can fit after 1 << 24 or so */

  RNN_NET_FLAG_STANDARD = (RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS \
       | RNN_COND_USE_ZERO | RNN_NET_FLAG_BIAS),
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

void rnn_multi_pgm_dump(RecurNN *net, const char *dumpees);

RecurNN* rnn_load_net(const char *filename);
int rnn_save_net(RecurNN *net, const char *filename);

void bptt_advance(RecurNN *net);
void bptt_calculate(RecurNN *net);

void bptt_consolidate_many_nets(RecurNN **nets, int n, int nestorov);

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
  ASSUME_ALIGNED(dest);
  ASSUME_ALIGNED(src);
  for (int i = 0; i < len; i++){
    dest[i] += src[i] * scale;
  }
#else
  cblas_saxpy(len, scale, src, 1, dest, 1);
#endif
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



#endif
