/* Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL/MPL2 */
#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include "badmaths.h"

static RecurNNBPTT *
new_bptt(RecurNN *net, int depth, float learn_rate, float momentum, u32 flags){
  RecurNNBPTT *bptt = calloc(sizeof(RecurNNBPTT), 1);
  int own_momentums = ! (flags & RNN_NET_FLAG_NO_MOMENTUMS);
  int own_deltas = ! (flags & RNN_NET_FLAG_NO_DELTAS);
  MAYBE_DEBUG("allocated bptt %p", bptt);
  bptt->depth = depth;
  bptt->learn_rate = learn_rate;
  bptt->momentum = momentum;
  bptt->momentum_weight = RNN_MOMENTUM_WEIGHT;
  size_t vlen = net->i_size * 2 + net->h_size * 0 + net->o_size * 1;
  if (own_deltas){
    vlen += net->ih_size * 2 + net->ho_size + 96;
  }
  if (own_momentums){
    vlen += net->ih_size + net->ho_size;
  }
  vlen += depth * net->i_size;

  float *fm = zalloc_aligned_or_die(vlen * sizeof(float));
  bptt->mem = fm;
  /*The haphazard arrangement of arrays is to avoid overly aligning the
    matrices, which has negative effects due to cache associativity*/
#define SET_ATTR_SIZE(attr, size) bptt->attr = fm; fm += (size)
  SET_ATTR_SIZE(o_error,           net->o_size);
  if (own_momentums){
    SET_ATTR_SIZE(ih_momentum,     net->ih_size);
    SET_ATTR_SIZE(ho_momentum,     net->ho_size);
  }
  /*h_error uses strictly larger i_size, facilitating switching between the 2*/
  SET_ATTR_SIZE(i_error,           net->i_size);
  SET_ATTR_SIZE(h_error,           net->i_size);
  if (own_deltas){
    /*just to be sure of cache-line misalignment */
    while ((((size_t)fm ^ (size_t)bptt->ih_momentum) & 0x3ff) == 0 ||
        (((size_t)fm ^ (size_t)net->ih_weights) & 0x3ff) == 0){
      fm += 32;
    }
    SET_ATTR_SIZE(ih_delta,        net->ih_size);
    SET_ATTR_SIZE(ho_delta,        net->ho_size);
  }
  SET_ATTR_SIZE(history,           depth * net->i_size);
  if (own_deltas){
    /*just to be sure of cache-line misalignment */
    if (((((size_t)fm ^ (size_t)bptt->ih_delta) & 0x3ff)) == 0){
      fm += 32;
    }
    SET_ATTR_SIZE(ih_delta_tmp,  net->ih_size);
  }

#undef SET_ATTR_SIZE
  MAYBE_DEBUG("allocated %lu floats, used %lu", vlen, fm - bptt->mem);

  bptt->index = 0;
  bptt->ho_scale = 1.0f;
  bptt->ih_scale = 1.0f;
  bptt->min_error_factor = BASE_MIN_ERROR_FACTOR * net->h_size;
  MAYBE_DEBUG("weights:   ih %p ho %p", net->ih_weights, net->ho_weights);
  MAYBE_DEBUG("momentum:  ih %p ho %p", bptt->ih_momentum, bptt->ho_momentum);
  MAYBE_DEBUG("delta:     ih %p ho %p", bptt->ih_delta, bptt->ho_delta);
  return bptt;
}

RecurNN *
rnn_new(uint input_size, uint hidden_size, uint output_size, u32 flags,
    u64 rng_seed, const char *log_file, int bptt_depth, float learn_rate,
    float momentum){
  RecurNN *net = calloc(1, sizeof(RecurNN));
  float *fm;
  /*sizes */
  size_t i_size = ALIGNED_VECTOR_LEN(hidden_size + input_size + 1, float);
  size_t h_size = ALIGNED_VECTOR_LEN(hidden_size + 1, float);
  size_t o_size = ALIGNED_VECTOR_LEN(output_size, float);
  size_t ih_size = i_size * h_size;
  size_t ho_size = h_size * o_size;
  /*scalar attributes */
  net->i_size = i_size;
  net->h_size = h_size;
  net->o_size = o_size;
  net->input_size = input_size;
  net->hidden_size = hidden_size;
  net->output_size = output_size;
  net->ih_size = ih_size;
  net->ho_size = ho_size;
  net->generation = 0;
  net->flags = flags;
  init_rand64_maybe_randomly(&net->rng, rng_seed);


  size_t alloc_bytes = (i_size + h_size + o_size) * sizeof(float);
  if (flags & RNN_NET_FLAG_OWN_WEIGHTS)
    alloc_bytes += (ih_size + ho_size) * sizeof(float);
  net->mem = fm = zalloc_aligned_or_die(alloc_bytes);

#define SET_ATTR_SIZE(attr, size) net->attr = fm; fm += (size)
  /*nodes and weights are shuffled around to (hopefully) break up page-wise
    alignment of co-accessed arrays (e.g. weights and momentums), because
    page-alignment allegedly works badly with the cache.
   */
  SET_ATTR_SIZE(input_layer, i_size);
  SET_ATTR_SIZE(hidden_layer, h_size);
  SET_ATTR_SIZE(output_layer, o_size);
  if (flags & RNN_NET_FLAG_OWN_WEIGHTS){
    SET_ATTR_SIZE(ih_weights, ih_size);
    SET_ATTR_SIZE(ho_weights, ho_size);
  }
#undef SET_ATTR_SIZE
  MAYBE_DEBUG("allocated %lu floats, used %lu", alloc_bytes / 4, fm - net->mem);
  MAYBE_DEBUG("flags is %d including bptt %d", flags, flags & RNN_NET_FLAG_OWN_BPTT);
  /* bptt */
  if (flags & RNN_NET_FLAG_OWN_BPTT){
    net->bptt = new_bptt(net, bptt_depth, learn_rate, momentum, flags);
    rnn_bptt_advance(net);
  }
  else {
    net->real_inputs = net->input_layer + net->hidden_size + 1;
  }

  if (log_file){
    rnn_set_log_file(net, log_file, flags & RNN_NET_FLAG_LOG_APPEND);
  }
  return net;
}

void
rnn_delete_net(RecurNN *net){
  if (net->bptt && (net->flags & RNN_NET_FLAG_OWN_BPTT)){
    free(net->bptt->mem);
    free(net->bptt);
  }
  if (net->log)
    fclose(net->log);
  free(net->mem);
  free(net);
}


RecurExtraLayer *
rnn_new_extra_layer(int input_size, int output_size, int overlap,
    u32 flags)
{
  RecurExtraLayer *layer = zalloc_aligned_or_die(sizeof(RecurExtraLayer));
  layer->input_size = input_size;
  layer->output_size = output_size;
  layer->input_size = input_size;
  layer->output_size = output_size;
  layer->overlap = overlap;
  layer->learn_rate_scale = 1.0;
  layer->i_size = ALIGNED_VECTOR_LEN(input_size + 1, float);
  layer->o_size = ALIGNED_VECTOR_LEN(output_size, float);
  int matrix_size = layer->i_size * layer->o_size;

  size_t floats = matrix_size * 3; /* weights, delta, momentums */
  floats += (layer->i_size + layer->o_size) * 2; /* nodes and errors */
  layer->mem = zalloc_aligned_or_die(floats * sizeof(float));
  float *fm = layer->mem;
#define SET_ATTR_SIZE(attr, size) do {layer->attr = fm; fm += (size);   \
    if (fm > layer->mem + floats) {                                     \
      DEBUG("Extra layer ran out of memory on " QUOTE(attr)             \
          " fm %p mem %p floats %zu allocating %d",                       \
          fm, layer->mem, floats, size);}} while(0)
  SET_ATTR_SIZE(momentums, matrix_size);
  SET_ATTR_SIZE(inputs, layer->i_size);
  SET_ATTR_SIZE(weights, matrix_size);
  SET_ATTR_SIZE(outputs, layer->o_size);
  SET_ATTR_SIZE(delta, matrix_size);
  SET_ATTR_SIZE(i_error, layer->i_size);
  SET_ATTR_SIZE(o_error, layer->o_size);
#undef SET_ATTR_SIZE
  return layer;
}

RecurNN *rnn_new_with_bottom_layer(int n_inputs, int r_input_size,
    int hidden_size, int output_size, u32 flags, u64 rng_seed,
    const char *log_file, int bptt_depth, float learn_rate,
    float momentum, int convolutional_overlap)
{
  RecurNN *net;
  if (r_input_size == 0){
    DEBUG("rnn_new_with_bottom_layer returning bottomless net, "
        "due to zero internal size");
    flags &= ~RNN_NET_FLAG_BOTTOM_LAYER;
    net = rnn_new(n_inputs, hidden_size, output_size,
        flags, rng_seed, log_file, bptt_depth, learn_rate, momentum);
  }
  else {
    flags |= RNN_NET_FLAG_BOTTOM_LAYER;
    net = rnn_new(r_input_size, hidden_size, output_size,
        flags, rng_seed, log_file, bptt_depth, learn_rate, momentum);

    net->bottom_layer = rnn_new_extra_layer(n_inputs, r_input_size,
        convolutional_overlap, net->flags);
  }
  return net;
}

RecurNN **
rnn_new_training_set(RecurNN *prototype, int n_nets){
  int i;
  if (n_nets < 1){ //XXX or is (n_nets < 2) better?
    DEBUG("A training set of size %u is not possible", n_nets);
    return NULL;
  }
  RecurNN **nets = malloc_aligned_or_die(n_nets * sizeof(RecurNN *));
  nets[0] = prototype;
  u32 flags = prototype->flags;

  flags &= ~RNN_NET_FLAG_OWN_WEIGHTS;
  flags |= RNN_NET_FLAG_NO_MOMENTUMS;
  flags |= RNN_NET_FLAG_NO_DELTAS;

  for (i = 1; i < n_nets; i++){
    nets[i] = rnn_clone(prototype, flags, RECUR_RNG_SUBSEED, NULL);
    nets[i]->bptt->ih_delta = prototype->bptt->ih_delta;
    nets[i]->bptt->ih_delta_tmp = prototype->bptt->ih_delta_tmp;
    nets[i]->bptt->ho_delta = prototype->bptt->ho_delta;
  }
  return nets;
}

void
rnn_delete_training_set(RecurNN** nets, int n_nets, int leave_prototype)
{
  /*If leave_prototype is true, the initial net that was not created by
    rnn_new_training_set() -- that is, nets[0] -- is not deleted.
  */
  int i = !! leave_prototype;
  for (; i < n_nets; i++){
    if (nets[i])
      rnn_delete_net(nets[i]);
  }
  free(nets);
}

/*to start logging, use
  rnn_set_log_file(net, "path-to-log-file", append_dont_truncate);

  If append_dont_truncate is true, the log file will be extended, otherwise it
  will be truncated and overwritten.

  To stop logging:
  rnn_set_log_file(net, NULL, 0);
 */
void
rnn_set_log_file(RecurNN *net, const char *log_file, int append_dont_truncate){
  if (net->log){
    fclose(net->log);
  }
  if (log_file){
    char *mode = append_dont_truncate ? "a" : "w";
    net->log = fopen(log_file, mode);
    if (! append_dont_truncate){
      rnn_log_int(net, "generation", net->generation);
    }
  }
  else{
    net->log = NULL;
  }
}

/*clone a net.
  if flags contains RNN_NET_FLAG_OWN_WEIGHTS,
  the net will have its own copy of the parent's weights.
  Otherwise it will borrow the parent's ones.

  if flags contains RNN_NET_FLAG_OWN_BPTT, the net will get a bptt training
  struct matching the parent's in structure but not inheriting momentum,
  history, and so forth.

 */

RecurNN *
rnn_clone(RecurNN *parent, u32 flags,
    u64 rng_seed, const char *log_file){
  RecurNN *net;
  if (rng_seed == RECUR_RNG_SUBSEED){
    do {
      rng_seed = rand64(&parent->rng);
    } /* -1 would lead to random seeding, drive debuggers crazy*/
    while(rng_seed == RECUR_RNG_RANDOM_SEED);
  }
  float learn_rate;
  int bptt_depth;
  float momentum;

  /*XXXX It would be nice to have a half bptt state -- with shared momentum
    but separate deltas, history */
  if (parent->bptt && (flags & RNN_NET_FLAG_OWN_BPTT)){
    learn_rate = parent->bptt->learn_rate;
    bptt_depth = parent->bptt->depth;
    momentum = parent->bptt->momentum;
  }
  else { /*doesn't matter what these are */
    learn_rate = 0;
    bptt_depth = 0;
    momentum = 0;
  }

  net = rnn_new(parent->input_size, parent->hidden_size, parent->output_size,
      flags, rng_seed, log_file, bptt_depth, learn_rate, momentum);

  if (parent->bptt && (flags & RNN_NET_FLAG_OWN_BPTT)){
    net->bptt->momentum_weight = parent->bptt->momentum_weight;
    if (flags & RNN_NET_FLAG_NO_MOMENTUMS){
      net->bptt->ih_momentum = parent->bptt->ih_momentum;
      net->bptt->ho_momentum = parent->bptt->ho_momentum;
    }
    if (flags & RNN_NET_FLAG_NO_DELTAS){
      net->bptt->ih_delta = parent->bptt->ih_delta;
      net->bptt->ho_delta = parent->bptt->ho_delta;
    }
  }
  if (flags & RNN_NET_FLAG_OWN_WEIGHTS){
    memcpy(net->ih_weights, parent->ih_weights, net->ih_size * sizeof(float));
    memcpy(net->ho_weights, parent->ho_weights, net->ho_size * sizeof(float));
  }
  else {
    net->ih_weights = parent->ih_weights;
    net->ho_weights = parent->ho_weights;
  }
  /*for now, the bottom layers can be shared */
  net->bottom_layer = parent->bottom_layer;
  net->generation = parent->generation;
  return net;
}

static inline void
maybe_randomise_using_submethod(RecurNN *net, struct RecurInitialisationParameters *p)
{
  if (p->submethod != p->method){
    /*randomise top layer and using submethod
      (presumably FLAT or FAN_IN)*/
    p->method = p->submethod;
    rnn_randomise_weights_clever(net, p);
    p->method = RNN_INIT_RUNS;
    STDERR_DEBUG("used submethod %d, bias too %d", p->submethod, p->bias_uses_submethod);
  }
  float *mem = net->ih_weights;
  size_t rows = p->inputs_use_submethod ? net->h_size : net->i_size;
  if (p->bias_uses_submethod){
    rows--;
    mem += net->h_size;
  }
  memset(mem, 0, rows * net->h_size * sizeof(float));
}


void
rnn_randomise_weights_clever(RecurNN *net, struct RecurInitialisationParameters *p){
  if (p->method == RNN_INIT_FAN_IN){
    rnn_randomise_weights_fan_in(net,
        p->fan_in_sum,
        p->fan_in_step,
        p->fan_in_min,
        p->fan_in_ratio);
  }
  else if (p->method == RNN_INIT_FLAT){
    rnn_randomise_weights_flat(net,
        p->flat_variance,
        p->flat_shape,
        p->flat_perforation
    );
  }
  else if (p->method == RNN_INIT_LOOPS){
    maybe_randomise_using_submethod(net, p);
    rnn_initialise_long_loops(net,
        p->loop_n,
        p->loop_len_mean,
        p->loop_len_stddev,
        p->loop_gain,
        p->loop_input_probability,
        p->loop_input_magnitude);
  }
  else if (p->method == RNN_INIT_RUNS){
    maybe_randomise_using_submethod(net, p);
    rnn_initialise_runs(net,
        p->runs_n,
        p->runs_gain,
        p->runs_min_gain,
        p->runs_input_magnitude,
        p->runs_avoid_loops);
  }
}

void
rnn_randomise_weights_auto(RecurNN *net){
  /*heuristically choose an initialisation scheme for the size of net */
  /*initial heuristic is very simple*/
  const rnn_init_method method = RNN_INIT_RUNS;
  //const rnn_init_method method = RNN_INIT_LOOPS;

  struct RecurInitialisationParameters p = {
    .method = method,
    .submethod = RNN_INIT_FLAT,
    .bias_uses_submethod = 0,
    .inputs_use_submethod = 0,

    .runs_n = 5 * net->h_size,
    .runs_gain = 0.12,
    .runs_min_gain = 1e-6,
    .runs_input_magnitude = 0.7,
    .runs_avoid_loops = 0,

    .fan_in_ratio = net->input_size  * 1.0f / net->hidden_size,
    .fan_in_sum = 3.0,
    .fan_in_step = 0.3,
    .fan_in_min = 0.1,

    .flat_variance = RNN_INITIAL_WEIGHT_VARIANCE_FACTOR / net->h_size,
    .flat_shape = 1,
    .flat_perforation = 0.0,

    .loop_input_probability = 0.12,
    .loop_input_magnitude = 0.4,
    .loop_gain = 0.17,
    .loop_len_mean = 7.0,
    .loop_len_stddev = 2.0,
    .loop_n = net->h_size * 3
  };
  rnn_randomise_weights_clever(net, &p);
}

static inline float
gaussian_power(rand_ctx *rng, float a, int power){
  for (int i = 0; i < power; i++){
    a *= cheap_gaussian_noise(rng);
  }
  return a;
}

void
rnn_randomise_weights_flat(RecurNN *net, float variance, int shape, double perforation){
  int x, y;
  memset(net->ih_weights, 0, net->ih_size * sizeof(float));
  memset(net->ho_weights, 0, net->ho_size * sizeof(float));
  /*higher shape indicates greater kurtosis. shape 0 means automatically
    determine shape (crudely).*/
  if (shape == 0){
    shape = (int)(1.0f + sqrtf(net->h_size / 400.0));
  }
  if (shape > 10){
    shape = 10;
  }
  if (perforation < 0){
    perforation = 0;
  }
  else if (perforation >= 1.0){
    return; /*perforation of 1 means entirely zeros */
  }
  for (y = 0; y < net->input_size + net->hidden_size + 1; y++){
    for (x = 1; x <= net->hidden_size; x++){
      if (rand_double(&net->rng) > perforation){
        net->ih_weights[y * net->h_size + x] = gaussian_power(&net->rng, variance, shape);
      }
    }
  }
  for (y = 0; y <= net->hidden_size; y++){
    for (x = 0; x < net->output_size; x++){
      if (rand_double(&net->rng) > perforation){
        net->ho_weights[y * net->o_size + x] = gaussian_power(&net->rng, variance, shape);
      }
    }
  }
  if (net->bottom_layer){
    RecurExtraLayer *bl = net->bottom_layer;
    memset(bl->weights, 0, bl->i_size * bl->o_size * sizeof(float));
    for (y = 0; y < bl->input_size; y++){
      for (x = 0; x < bl->output_size; x++){
        if (rand_double(&net->rng) > perforation){
          bl->weights[y * bl->o_size + x] = gaussian_power(&net->rng,
              variance, shape);
        }
      }
    }
  }
}

static inline void
randomise_weights_fan_in(rand_ctx *rng, float *weights, int width, int height, int stride,
    float sum, float kurtosis, float margin){
  int x, y, i;
  /*Each node gets input that adds, approximately, to <sum> */
  for (x = 0; x < width; x++){
    float remainder = sum + margin;
    for (i = 0; i < height * 2 && remainder > margin; i++){
      y = rand_small_int(rng, height);
      if (weights[y * stride + x] == 0){
        float w = (rand_double(rng) * 2 - 1) * remainder * kurtosis;
        weights[y * stride + x] += w;
        remainder -= fabsf(w);
      }
    }
  }
}

void
rnn_randomise_weights_fan_in(RecurNN *net, float sum, float kurtosis,
    float margin, float inputs_weight_ratio){
  memset(net->ih_weights, 0, net->ih_size * sizeof(float));
  memset(net->ho_weights, 0, net->ho_size * sizeof(float));
  int hsize = 1 + net->hidden_size;
  if (inputs_weight_ratio > 0){
    randomise_weights_fan_in(&net->rng, net->ih_weights + 1,
        net->hidden_size, hsize, net->h_size, sum, kurtosis, margin);
    randomise_weights_fan_in(&net->rng, net->ih_weights + hsize * net->h_size + 1,
        net->hidden_size, net->input_size, net->h_size,
        sum * inputs_weight_ratio, kurtosis, margin);
  }
  else {
    randomise_weights_fan_in(&net->rng, net->ih_weights + 1,
        net->hidden_size, hsize + net->input_size, net->h_size,
        sum, kurtosis, margin);
  }
  randomise_weights_fan_in(&net->rng, net->ho_weights, net->output_size, net->hidden_size,
      net->o_size, sum, kurtosis, margin);

  if (net->bottom_layer){
    RecurExtraLayer *bl = net->bottom_layer;
    memset(bl->weights, 0, bl->i_size * bl->o_size * sizeof(float));
    randomise_weights_fan_in(&net->rng, bl->weights, bl->output_size,
        bl->input_size + 1, bl->o_size,
        sum, kurtosis, margin);
  }
}


void rnn_perforate_weights(RecurNN *net, float p){
  dropout_array(net->ih_weights, net->ih_size, p, &net->rng);
  dropout_array(net->ho_weights, net->ho_size, p, &net->rng);
}

void rnn_emphasise_diagonal(RecurNN *net, float magnitude, float proportion){
  int i;
  int n = MIN(net->hidden_size * proportion + 1, net->hidden_size);

  for (i = 1; i < n; i++){
    int offset = i * (net->h_size + 1);
    net->ih_weights[offset] += rand_double(&net->rng) * 2 * magnitude - magnitude;
  }
}

static inline float
bounded_log_normal_random_sign(rand_ctx *rng, float mean, float stddev, float bound)
{
  /*stddev is std deviation in log space; bound is standard deviations.*/
  float x, w;
  do {
    x = cheap_gaussian_noise(rng);
  } while(fabsf(x) > bound);
  w = mean * fast_expf(x * stddev);
  return (rand64(rng) & 1) ? w : -w;
}

static inline int
weight_run(RecurNN *net, const float gain_per_step,
    const float min_gain, float const input_magnitude, int avoid_loops){
  int i, j, s, e;
  float weight;
  float gain = 1.0f;

  int unused[net->hidden_size + 1];

  e = RAND_SMALL_INT_RANGE(&net->rng, 0, net->hidden_size) + 1;
  if (avoid_loops){
    j = e;
    for (i = 0; i <= net->hidden_size; i++){
      unused[i] = i;
    }
  }

  if (input_magnitude) {
    int input = RAND_SMALL_INT_RANGE(&net->rng, 0, net->input_size);
    net->ih_weights[(net->hidden_size + 1 + input) * net->h_size + e] = \
      cheap_gaussian_noise(&net->rng) * input_magnitude;
  }

  for(i = 1; i <= net->hidden_size; i++){
    s = e;
    if (avoid_loops){
      unused[j] = unused[i];
      j = RAND_SMALL_INT_RANGE(&net->rng, i, net->hidden_size) + 1;
      e = unused[j];
    }
    else {
      e = RAND_SMALL_INT_RANGE(&net->rng, 0, net->hidden_size) + 1;
    }
    weight = bounded_log_normal_random_sign(&net->rng, gain_per_step, 0.25, 3.0);
    gain *= fabsf(weight);
    net->ih_weights[s * net->h_size + e] = weight;
    if (gain < min_gain){
      break;
    }
  }
  return i;
}

void
rnn_initialise_runs(RecurNN* net, int n_runs, float gain_per_step,
    float min_gain, float input_magnitude, int avoid_loops){
  STDERR_DEBUG("n_runs %d gain_per_step %g, min_gain %g, "
      "input_magnitude %g avoid_loops %d",
      n_runs, gain_per_step, min_gain, input_magnitude, avoid_loops);
  int len = 0;
  for (int i = 0; i < n_runs; i++){
    len += weight_run(net, gain_per_step, min_gain, input_magnitude, avoid_loops);
  }
  STDERR_DEBUG("mean run length %2g", len / (double)n_runs);
}


/*loop_link is a helper for long_loop().
  It connects the hidden node <s> to the hidden node <e>, with
  approximately <gain> strength, and maybe adds in an input.
*/

static inline void
loop_link(RecurNN *net, int s, int e,
    const float gain, const float input_probability,
    float const input_magnitude){
  float weight = bounded_log_normal_random_sign(&net->rng, gain, 0.25, 3.0);
  net->ih_weights[s * net->h_size + e] = weight;
  if (rand_double(&net->rng) < input_probability){
    int input = RAND_SMALL_INT_RANGE(&net->rng, 0, net->input_size);
    net->ih_weights[(net->hidden_size + 1 + input) * net->h_size + e] = \
      cheap_gaussian_noise(&net->rng) * input_magnitude;
  }
}

static inline void
long_loop(RecurNN *net, int len, const float gain_per_step,
    const float input_probability, float const input_magnitude){
  int i, j, s, e, beginning;
  len = MIN(len, net->hidden_size);
  /*unused is a one-based list of unused nodes. At step i, unused[i+1:]
    contains all the previously unused nodes. This is the same as a basic
    shuffle algorithm, but there is no need to save the already shuffled
    nodes.*/
  int unused[net->hidden_size + 1];
  for (i = 0; i <= net->hidden_size; i++){
    unused[i] = i;
  }
  j = RAND_SMALL_INT_RANGE(&net->rng, 0, net->hidden_size) + 1;
  beginning = e = j;

  for (i = 1; i <= len; i++){
    unused[j] = unused[i];
    s = e;
    j = RAND_SMALL_INT_RANGE(&net->rng, i, net->hidden_size) + 1;
    e = unused[j];
    loop_link(net, s, e, gain_per_step, input_probability, input_magnitude);
  }
  /*now to complete the loop, link back to the beginning.*/
  loop_link(net, e, beginning, gain_per_step, input_probability, input_magnitude);
}


void
rnn_initialise_long_loops(RecurNN* net, int n_loops, int len_mean, int len_stddev,
    float gain, float input_probability, float input_magnitude){
  STDERR_DEBUG("n_loops %d len_mean %d, len_stddev %d, gain %g, "
      "input_probability %g, input_magnitude %g",
      n_loops, len_mean, len_stddev, gain, input_probability, input_magnitude);
  int sum = 0;
  for (int i = 0; i < n_loops; i++){
    int len = cheap_gaussian_noise(&net->rng) * len_stddev + len_mean + 0.5;
    len = MAX(2, len);
    sum += len;
    long_loop(net, len, gain, input_probability, input_magnitude);
  }
  STDERR_DEBUG("mean loop len %3g", (double)sum / n_loops);
}

void
rnn_scale_initial_weights(RecurNN *net, float factor){
  float ih_sum = abs_sum_aligned_array(net->ih_weights, net->ih_size);
  float ho_sum = abs_sum_aligned_array(net->ho_weights, net->ho_size);

  factor *= (net->i_size + net->h_size + net->o_size);
  float ih_ratio = sqrtf(factor * (net->h_size * net->i_size)) / ih_sum;
  float ho_ratio = sqrtf(factor * (net->h_size * net->o_size)) / ho_sum;

  scale_aligned_array(net->ih_weights, net->ih_size, ih_ratio);
  scale_aligned_array(net->ho_weights, net->ho_size, ho_ratio);

  DEBUG("ih sum was %.1f, now %.1f; ratio %.2g",
      ih_sum, abs_sum_aligned_array(net->ih_weights, net->ih_size),
      ih_ratio);

  DEBUG("ho sum was %.1f, now %.1f; ratio %.2g",
      ho_sum, abs_sum_aligned_array(net->ho_weights, net->ho_size),
      ho_ratio);
}

void
rnn_multi_pgm_dump(RecurNN *net, const char *dumpees, const char *basename){
  RecurNNBPTT *bptt = net->bptt;
  char *working = strdupa(dumpees);
  char *token;
  while ((token = strsep(&working, " "))){
    int x = 0, y = 0;
    float *array = NULL;
    if (strlen(token) != 3)
      continue;
    char in = token[0];
    char out = token[1];
    char v = token[2];
    if (out == 'h'){
      x = net->h_size;
      if (in == 'i')
        y = net->i_size;
      else if (in == 'h')
        y = net->hidden_size;
      else
        continue;
      if (v == 'w')
        array = net->ih_weights;
      else if (v == 'm')
        array = bptt->ih_momentum;
      else if (v == 'd')
        array = bptt->ih_delta;
      else if (v == 't')
        array = bptt->ih_delta_tmp;
      else
        continue;
    }
    else if (in == 'h' && out == 'o'){
      x = net->o_size;
      y = net->h_size;
      if (v == 'w')
        array = net->ho_weights;
      else if (v == 'm')
        array = bptt->ho_momentum;
      else if (v == 'd')
        array = bptt->ho_delta;
      else
        continue;
    }
    else if (in == 'b' && out == 'i'){
      RecurExtraLayer *b = net->bottom_layer;
      if (!b)
        continue;
      x = b->o_size;
      y = b->i_size;
      if (v == 'w')
        array = b->weights;
      else if (v == 'm')
        array = b->momentums;
      else if (v == 'd')
        array = b->delta;
      else
        continue;
    }
    if (array){
      if (! basename || ! basename[0]){
        basename = "untitled";
      }
      char *name;
      if (asprintf(&name, "%s-%s", basename, token) >= 0){
        dump_colour_weights_autoname(array, x, y, name, net->generation);
        free(name);
      }
      else {
        STDERR_DEBUG("Can't asprintf in rnn_multi_pgm_dump; not dumping");
      }
    }
  }
}
