#include "recur-nn.h"
#include "recur-nn-helpers.h"

static RecurNNBPTT *
new_bptt(RecurNN *net, int depth, float learn_rate, float momentum, u32 flags){
  RecurNNBPTT *bptt = calloc(sizeof(RecurNNBPTT), 1);
  int own_momentums = ! (flags & RNN_NET_FLAG_NO_MOMENTUMS);
  int own_deltas = ! (flags & RNN_NET_FLAG_NO_DELTAS);
  int own_accumulators = flags & RNN_NET_FLAG_OWN_ACCUMULATORS;
  MAYBE_DEBUG("allocated bptt %p", bptt);
  bptt->depth = depth;
  bptt->learn_rate = learn_rate;
  bptt->momentum = momentum;
  bptt->momentum_weight = RNN_MOMENTUM_WEIGHT;
  size_t vlen = net->i_size * 2 + net->h_size * 0 + net->o_size * 1;
  if (own_deltas){
    vlen += net->ih_size + net->ho_size + 32;
  }
  if (own_accumulators){
    vlen += net->ih_size + net->ho_size + 64;
  }
  if (own_momentums){
    vlen += net->ih_size + net->ho_size;
  }
  vlen += depth * net->i_size;

  float *fm = zalloc_aligned_or_die(vlen * sizeof(float));
  bptt->mem = fm;
  /*The haphazard arrangement of arrays is to avoid overly aligning the
    matrices, which has negative effects due to cache associativity*/
#define SET_ATTR_SIZE(attr, size) bptt->attr = fm; fm += (size);
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
    if (((((size_t)fm ^ (size_t)bptt->ih_momentum) & 0x3ff)) == 0){
      fm += 32;
    }
    SET_ATTR_SIZE(ih_delta,        net->ih_size);
    SET_ATTR_SIZE(ho_delta,        net->ho_size);
  }
  SET_ATTR_SIZE(history,           depth * net->i_size);
  if (own_accumulators){
    /*just to be sure of cache-line misalignment */
    while ((((size_t)fm ^ (size_t)bptt->ih_delta) & 0x3ff) == 0 ||
        (((size_t)fm ^ (size_t)net->ih_weights) & 0x3ff) == 0){
      fm += 32;
    }
    SET_ATTR_SIZE(ih_accumulator,  net->ih_size);
    SET_ATTR_SIZE(ho_accumulator,  net->ho_size);
  }

#undef SET_ATTR_SIZE
  MAYBE_DEBUG("allocated %lu floats, used %lu", vlen, fm - bptt->mem);

  bptt->index = 0;
  bptt->ho_scale = sqrtf(((float) net->output_size) / net->hidden_size);
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
  int bias = !! (flags & RNN_NET_FLAG_BIAS);
  float *fm;
  /*sizes */
  size_t i_size = ALIGNED_VECTOR_LEN(hidden_size + input_size + bias, float);
  size_t h_size = ALIGNED_VECTOR_LEN(hidden_size + bias, float);
  size_t o_size = ALIGNED_VECTOR_LEN(output_size, float);
  size_t ih_size = i_size * h_size;
  size_t ho_size = h_size * o_size;
  /*scalar attributes */
  net->bias = bias;
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

#define SET_ATTR_SIZE(attr, size) net->attr = fm; fm += (size);
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
    net->real_inputs = net->input_layer + net->hidden_size + net->bias;
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

RecurNN **
rnn_new_training_set(RecurNN *prototype, int n_nets){
  int i;
  if (n_nets < 1){ //XXX or is (n_nets < 2) better?
    DEBUG("A training set of size %u is not possible", n_nets);
    return NULL;
  }
  if (! prototype->bptt->ih_accumulator ||
      ! prototype->bptt->ho_accumulator){
    DEBUG("The training set prototype lacks delta accumulators");
    DEBUG("Adding delta accumulators. This memory will not be reclaimed!");
    DEBUG("Also adding the RNN_NET_FLAG_OWN_ACCUMULATORS flag,");
    DEBUG("so everything should be right after the next save/load cycle");
    size_t accum_size = (prototype->ih_size + prototype->ho_size + 64) * sizeof(float);
    float *fm = malloc_aligned_or_die(accum_size);
    while ((((size_t)fm ^ (size_t)prototype->bptt->ih_delta) & 0x3ff) == 0 ||
        (((size_t)fm ^ (size_t)prototype->ih_weights) & 0x3ff) == 0){
      fm += 32;
    }
    prototype->bptt->ih_accumulator = fm;
    prototype->bptt->ho_accumulator = fm + prototype->ih_size;
    prototype->flags |= RNN_NET_FLAG_OWN_ACCUMULATORS;
  }

  RecurNN **nets = malloc_aligned_or_die(n_nets * sizeof(RecurNN *));
  nets[0] = prototype;
  u32 flags = prototype->flags;

  flags &= ~RNN_NET_FLAG_OWN_WEIGHTS;
  flags &= ~RNN_NET_FLAG_OWN_ACCUMULATORS;
  flags |= RNN_NET_FLAG_NO_MOMENTUMS;
  flags |= RNN_NET_FLAG_NO_DELTAS;

  for (i = 1; i < n_nets; i++){
    nets[i] = rnn_clone(prototype, flags, RECUR_RNG_SUBSEED, NULL);
    nets[i]->bptt->ih_delta = prototype->bptt->ih_delta;
    nets[i]->bptt->ho_delta = prototype->bptt->ho_delta;
    nets[i]->bptt->ih_accumulator = prototype->bptt->ih_accumulator;
    nets[i]->bptt->ho_accumulator = prototype->bptt->ho_accumulator;
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

  the RNN_NET_FLAG_BIAS has no effect: the parent net's bias is used.
 */

RecurNN *
rnn_clone(RecurNN *parent, u32 flags,
    u64 rng_seed, const char *log_file){
  RecurNN *net;
  if (parent->bias)
    flags |= RNN_NET_FLAG_BIAS;
  else
    flags &= ~RNN_NET_FLAG_BIAS;

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
  net->generation = parent->generation;
  return net;
}


void
rnn_randomise_weights_auto(RecurNN *net){
  /*heuristically choose an initialisation scheme for the size of net */
  /*initial heuristic is very simple*/
  float ratio = net->input_size  * 1.0f / net->hidden_size;
  rnn_randomise_weights_fan_in(net, 2.0, 0.3, 0.1, ratio);
}

static inline float
gaussian_power(rand_ctx *rng, float a, int power){
  for (int i = 0; i < power; i++){
    a *= cheap_gaussian_noise(rng);
  }
  return a;
}

void
rnn_randomise_weights(RecurNN *net, float variance, int shape, double perforation){
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
  for (y = 0; y < net->i_size; y++){
    for (x = net->bias; x < net->hidden_size + net->bias; x++){
      if (rand_double(&net->rng) > perforation){
        net->ih_weights[y * net->h_size + x] = gaussian_power(&net->rng, variance, shape);
      }
    }
  }
  for (y = 0; y < net->h_size; y++){
    for (x = 0; x < net->output_size; x++){
      if (rand_double(&net->rng) > perforation){
        net->ho_weights[y * net->o_size + x] = gaussian_power(&net->rng, variance, shape);
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

  int hsize = net->bias + net->hidden_size;
  if (inputs_weight_ratio > 0){
    randomise_weights_fan_in(&net->rng, net->ih_weights + net->bias,
        net->hidden_size, hsize, net->h_size, sum, kurtosis, margin);
    randomise_weights_fan_in(&net->rng, net->ih_weights + hsize * net->h_size + net->bias,
        net->hidden_size,
        net->input_size, net->h_size, sum * inputs_weight_ratio, kurtosis, margin);
  }
  else {
    randomise_weights_fan_in(&net->rng, net->ih_weights + net->bias,
        net->hidden_size, hsize + net->input_size, net->hidden_size,
        sum, kurtosis, margin);
  }
  randomise_weights_fan_in(&net->rng, net->ho_weights, net->output_size, net->hidden_size,
      net->o_size, sum, kurtosis, margin);
}

void rnn_perforate_weights(RecurNN *net, float p){
  dropout_array(net->ih_weights, net->ih_size, p, &net->rng);
  dropout_array(net->ho_weights, net->ho_size, p, &net->rng);
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
rnn_multi_pgm_dump(RecurNN *net, const char *dumpees){
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
    if (array)
      dump_colour_weights_autoname(array, x, y, token, net->generation);
  }
}
