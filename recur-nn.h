/* Copyright 2013 Douglas Bagnall <douglas@halo.gen.nz> LGPL */

#ifndef _GOT_RECUR_NN_H
#define _GOT_RECUR_NN_H 1

#include <unistd.h>
#include <string.h>
#include "recur-common.h"
#include "recur-rng.h"

/*VECTOR and VECTOR_ALL_THE_WAY now set in Makefile or local.mak */
//#define VECTOR 1
//#define VECTOR_ALL_THE_WAY (0 && VECTOR)

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
/*if bptt error grows by more than MAX_ERROR_GAIN, abort the bptt loop */
#define MAX_ERROR_GAIN 2.0f
/*if final bptt error is greater than than ERROR_GAIN_CEILING, scale it down */
#define ERROR_GAIN_CEILING 1.0f
/*BASE_MIN_ERROR_FACTOR relates to initial minimum mean error*/
#define BASE_MIN_ERROR_FACTOR 1e-12f
/*MAX_MIN_ERROR_FACTOR puts a limit on the growth of min_error_factor*/
#define MAX_MIN_ERROR_FACTOR 1e-2f
/*min_error_factor never goes below ABS_MIN_ERROR_FACTOR*/
#define ABS_MIN_ERROR_FACTOR 1e-20f
/*MIN_ERROR_GAIN */
#define MIN_ERROR_GAIN 1e-8f
/* RNN_HIDDEN_PENALTY is subtracted from each hidden node, forcing low numbers to zero.
 1e-3f is safe, but less accurate than 1e-4f
XXX this really ought to be adjustable or adjust itself */
#define RNN_HIDDEN_PENALTY 0.0f
/*scaling for hidden and input numbers */
#define HIDDEN_MEAN_SOFT_TOP 16.0f
#define INPUT_MEAN_SOFT_TOP 16.0f

#define RNN_INITIAL_WEIGHT_VARIANCE_FACTOR 2.0f
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
  //RNN_NET_FLAG_BIAS = 4, /*reserved for a while */
  RNN_NET_FLAG_LOG_APPEND = 8,
  RNN_NET_FLAG_LOG_HIDDEN_SUM = 16, /*log the hidden sum */
  RNN_NET_FLAG_LOG_WEIGHT_SUM = 32, /*log the weight sum (can be expensive)*/
  RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR = 64, /*min error threshold auto-adjusts*/
  RNN_NET_FLAG_NO_MOMENTUMS = 128, /*allocate no momentum arrays (borrow parent's)*/
  RNN_NET_FLAG_NO_DELTAS = 256, /* allocated no delta array (borrow parent's)*/
  /*XXX accumulators flag is gone */
  RNN_NET_FLAG_BOTTOM_LAYER = 1024, /*network has a layer below RNN*/
  RNN_NET_FLAG_AUX_ARRAYS = 2048, /*allocate an extra training array (adadelta, etc)*/

  /*conditioning flags start at 1 << 16 (65536) */
  RNN_COND_USE_SCALE = (1 << (RNN_COND_BIT_SCALE + RNN_COND_USE_OFFSET)),
  RNN_COND_USE_ZERO = (1 << (RNN_COND_BIT_ZERO + RNN_COND_USE_OFFSET)),
  RNN_COND_USE_LAWN_MOWER = (1 << (RNN_COND_BIT_LAWN_MOWER + RNN_COND_USE_OFFSET)),
  RNN_COND_USE_TALL_POPPY = (1 << (RNN_COND_BIT_TALL_POPPY + RNN_COND_USE_OFFSET)),
  RNN_COND_USE_RAND = (1 << (RNN_COND_BIT_RAND + RNN_COND_USE_OFFSET)),

  /*more flags can fit after 1 << 24 or so */

  RNN_NET_FLAG_STANDARD = (RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS \
       | RNN_COND_USE_ZERO | RNN_NET_FLAG_LOG_HIDDEN_SUM),
};


/*initial momentum weight for weighted momentum */
#define RNN_MOMENTUM_WEIGHT 0.5f

typedef enum {
  RNN_MOMENTUM_WEIGHTED = 0,
  RNN_MOMENTUM_NESTEROV,
  RNN_MOMENTUM_SIMPLIFIED_NESTEROV,
  RNN_MOMENTUM_CLASSICAL,
  RNN_ADAGRAD,
  RNN_ADADELTA,
  RNN_RPROP,

  RNN_LAST_LEARNING_METHOD
} rnn_learning_method;

typedef enum {
  RNN_INIT_FLAT = 1,
  RNN_INIT_FAN_IN,
  RNN_INIT_RUNS,

  RNN_INIT_LAST
} rnn_init_method;

typedef enum {
  RNN_RELU = 1,
  RNN_RESQRT,
  RNN_RELOG,
  RNN_RETANH,
  RNN_RECLIP20,

  RNN_ACTIVATION_LAST
} rnn_activation;

typedef enum {
  /*if you change this, also change text-predict's
    --flat-init-distribution documentation*/
  RNN_INIT_DIST_UNIFORM = 1,
  RNN_INIT_DIST_GAUSSIAN,
  RNN_INIT_DIST_LOG_NORMAL,
  RNN_INIT_DIST_SEMICIRCLE,

  RNN_INIT_DIST_DEFAULT
} rnn_init_distribution;


typedef struct _RecurNN RecurNN;
typedef struct _RecurNNBPTT RecurNNBPTT;
typedef struct _RecurExtraLayer RecurExtraLayer;

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
  RecurExtraLayer *bottom_layer;
  char *metadata;
  u32 generation;
  float presynaptic_noise;
  rnn_activation activation;
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
  float *ih_delta_tmp;
  float *ih_aux;
  float *ho_aux;
  float *mem;
  float learn_rate;
  float ih_scale;
  float ho_scale;
  float momentum;
  float momentum_weight;
  float min_error_factor;
};

struct _RecurExtraLayer {
  float *mem;
  float *weights;
  float *momentums;
  float *aux;
  float *delta;
  float *inputs;
  float *outputs;
  float *i_error;
  float *o_error;
  float learn_rate_scale;
  int input_size;
  int output_size;
  int i_size;
  int o_size;
  int overlap;
};


struct RecurInitialisationParameters {
  rnn_init_method method;
  rnn_init_method submethod;
  int bias_uses_submethod;
  int inputs_use_submethod;

  /*fan in */
  float fan_in_sum;
  float fan_in_step;
  float fan_in_min;
  float fan_in_ratio;

  /*flat */
  float flat_variance;
  rnn_init_distribution flat_shape;
  double flat_perforation;

  /* runs */
  float run_input_probability;
  float run_input_magnitude;
  float run_gain;
  float run_len_mean;
  float run_len_stddev;
  int run_n;
  int run_loop;
  int run_crossing_paths;
  int run_inputs_miss;
  int run_input_at_start;
};

typedef struct _RecurErrorRange RecurErrorRange;

struct _RecurErrorRange {
  int start;
  int len;
};

/* functions */

RecurNN * rnn_new(uint input_size, uint hidden_size, uint output_size,
    u32 flags, u64 rng_seed, const char *log_file, int depth, float learn_rate,
    float momentum, float presynaptic_noise, rnn_activation activation);

RecurNN * rnn_clone(RecurNN *parent, u32 flags, u64 rng_seed, const char *log_file);

RecurExtraLayer *rnn_new_extra_layer(int input_size, int output_size, int overlap,
    u32 flags);

RecurNN *rnn_new_with_bottom_layer(int n_inputs, int r_input_size,
    int hidden_size, int output_size, u32 flags, u64 rng_seed,
    const char *log_file, int bptt_depth, float learn_rate,
    float momentum, float presynaptic_noise,
    rnn_activation activation, int convolutional_overlap);


void rnn_set_log_file(RecurNN *net, const char * log_file, int append_dont_truncate);

void rnn_randomise_weights_clever(RecurNN *net, struct RecurInitialisationParameters *p);
void rnn_randomise_weights_simple(RecurNN *net, const rnn_init_method method);
void rnn_randomise_weights_auto(RecurNN *net);

void rnn_init_default_weight_parameters(RecurNN *net,
    struct RecurInitialisationParameters *q);

void rnn_scale_initial_weights(RecurNN *net, float target_gain);

void rnn_print_net_stats(RecurNN *net);

void rnn_delete_net(RecurNN *net);
RecurNN ** rnn_new_training_set(RecurNN *prototype, int n_nets);
void rnn_delete_training_set(RecurNN **nets, int n_nets, int leave_prototype);

float *rnn_opinion(RecurNN *net, const float *inputs, float presynaptic_noise);

void rnn_multi_pgm_dump(RecurNN *net, const char *dumpees, const char *basename);

RecurNN* rnn_load_net(const char *filename);
int rnn_save_net(RecurNN *net, const char *filename, int backup);

void rnn_bptt_clear_deltas(RecurNN *net);
void rnn_bptt_advance(RecurNN *net);
void rnn_bptt_calculate(RecurNN *net, uint batch_size);
void rnn_apply_learning(RecurNN *net, int learning_style, float momentum);
float rnn_calculate_momentum_soft_start(float generation, float momentum,
    float momentum_soft_start);

void rnn_bptt_calc_deltas(RecurNN *net, int accumulate_delta,
    RecurErrorRange *top_error_ranges);

void rnn_condition_net(RecurNN *net);
void rnn_log_net(RecurNN *net);

void rnn_forget_history(RecurNN *net, int bptt_too);

void rnn_perforate_weights(RecurNN *net, float p);

void rnn_apply_extra_layer_learning(RecurExtraLayer *layer);

void rnn_weight_noise(RecurNN *net, float deviation);

void rnn_set_momentum_values(RecurNN *net, float x);
void rnn_set_aux_values(RecurNN *net, float x);


static inline void
rnn_log_float(RecurNN *net, char *name, float value){
  if (net->log){
    fprintf(net->log, "%s %.5g\n", name, value);
  }
}

static inline void
rnn_log_int(RecurNN *net, char *name, int value){
  if (net->log){
    fprintf(net->log, "%s %d\n", name, value);
  }
}


#endif
