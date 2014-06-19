/* Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL */
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
    MAYBE_DEBUG("rnn_new_with_bottom_layer returning bottomless net, "
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



/* Weight initialisation */

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

/*link a random input node to a given hidden node */
static inline void
add_random_input(RecurNN *net, int dest, float deviation){
  int input = RAND_SMALL_INT_RANGE(&net->rng, 0, net->input_size);
  net->ih_weights[(net->hidden_size + 1 + input) * net->h_size + dest] =   \
    cheap_gaussian_noise(&net->rng) * deviation;
}

/*loop_link is a helper for initialise_loops_or_runs().
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
    add_random_input(net, e, input_magnitude);
  }
}

static void
initialise_loops_or_runs(RecurNN* net, int n_loops, int len_mean, int len_stddev,
    float gain, float input_probability, float input_magnitude,
    int loop, int crossing_paths, int inputs_miss, int input_at_start){
  STDERR_DEBUG("n_loops %d len_mean %d, len_stddev %d, gain %g, "
      "input_probability %g, input_magnitude %g loop %d crossing_paths %d,"
      " inputs_miss %d input_at_start %d",
      n_loops, len_mean, len_stddev, gain, input_probability, input_magnitude,
      loop, crossing_paths, inputs_miss, input_at_start);

  int sum = 0;
  int beginning, e, s, j;
  int bound = net->hidden_size + 1;
  int i = bound;
  int unused[bound];

  double linked_input_p =  inputs_miss ? 0 : input_probability;
  double missing_input_p = inputs_miss ? input_probability : 0;

  for (int loop_count = 0; loop_count < n_loops; loop_count++){
    int len = cheap_gaussian_noise(&net->rng) * len_stddev + len_mean + 0.5;
    len = MIN(MAX(2, len), net->hidden_size);

    /* if it doesn't fit, or if non-exclusive paths have been requested, reset
       the unused list. Always happens first round.

       In the crossing_paths == 2 case, this work is unnecessary (but harmless).
    */
    if (i + len + inputs_miss >= bound || crossing_paths){
      for (int k = 0; k < bound; k++){
        unused[k] = k;
      }
      i = 1;
    }

    j = RAND_SMALL_INT_RANGE(&net->rng, i, bound);
    beginning = e = unused[j];

    if (input_at_start && input_magnitude) {
      add_random_input(net, e, input_magnitude);
    }

    for (int m = 0; m < len; m++, i++){
      unused[j] = unused[i];
      s = e;
      /*crossing_paths == 2 means path can self-cross*/
      if (crossing_paths == 2){
        e = RAND_SMALL_INT_RANGE(&net->rng, 1, bound);
      }
      else {
        j = RAND_SMALL_INT_RANGE(&net->rng, i, bound);
        e = unused[j];
      }
      loop_link(net, s, e, gain, linked_input_p, input_magnitude);
    }

    if (loop){
      /*loop the end to the beginning */
      loop_link(net, e, beginning, gain, linked_input_p, input_magnitude);
    }

    /* missing inputs, if they are wanted, land randomly outside of loops.*/
    if (rand_double(&net->rng) < missing_input_p && i < bound){
      j = RAND_SMALL_INT_RANGE(&net->rng, i, bound);
      e = unused[j];
      unused[j] = unused[i];
      i++;
      add_random_input(net, e, input_magnitude);
    }
    sum += len;
  }
  STDERR_DEBUG("mean loop len %3g", (double)sum / n_loops);
}



static inline void
randomise_array_flat(rand_ctx *rng, float *array,
    const int width, const int height, const int stride,
    const int offset, const float variance,
    const rnn_init_distribution shape, const double perforation){
  int x, y;
  float stddev = sqrtf(variance);
  STDERR_DEBUG("using method %d, variance %g", shape, variance);
  for (y = 0; y < height; y++){
    for (x = offset; x < width + offset; x++){
      if (perforation == 0 ||
          rand_double(rng) > perforation){
        switch (shape){
        case RNN_INIT_DIST_UNIFORM:
          {
            const double range = sqrtf(12.0f * variance);
            array[y * stride + x] = range * rand_double(rng) - range * 0.5;
          }
          break;

        default:
        case RNN_INIT_DIST_GAUSSIAN:
          array[y * stride + x] = stddev * cheap_gaussian_noise(rng);
          break;

        case RNN_INIT_DIST_LOG_NORMAL:
          {
            /*XXX variance is doing 2 things*/
            float a = cheap_gaussian_noise(rng) * 0.33;
            float b = 0.9 * stddev * fast_expf(a);
            array[y * stride + x] = (rand64(rng) & 1) ? b : -b;
          }
          break;

        case RNN_INIT_DIST_SEMICIRCLE:
          {
            /*sample from the square until we hit the circle.
             variance = r^2/4 -- r = 4v */
            double a, b;
            do {
              a = rand_double(rng) * 2.0 - 1.0;
              b = rand_double(rng);
            } while (a * a + b * b > 1.0);
            array[y * stride + x] = stddev * 2 * a;
          }
          break;
        }
      }
    }
  }
}

static void
randomise_weights_flat(RecurNN *net, float variance,
    rnn_init_distribution shape, double perforation){
  memset(net->ih_weights, 0, net->ih_size * sizeof(float));
  memset(net->ho_weights, 0, net->ho_size * sizeof(float));
  if (perforation < 0){
    perforation = 0;
  }
  else if (perforation >= 1.0){
    return; /*perforation of 1 means entirely zeros */
  }
  randomise_array_flat(&net->rng, net->ih_weights,
      net->hidden_size, net->input_size + net->hidden_size + 1, net->h_size,
      1, variance, shape, perforation);

  randomise_array_flat(&net->rng, net->ho_weights,
      net->output_size, net->hidden_size + 1, net->o_size,
      0, variance, shape, perforation);

  if (net->bottom_layer){
    RecurExtraLayer *bl = net->bottom_layer;
    memset(bl->weights, 0, bl->i_size * bl->o_size * sizeof(float));
    randomise_array_flat(&net->rng, bl->weights,
        bl->output_size, bl->input_size, bl->o_size,
        1, variance, shape, perforation);
  }
}

static inline void
randomise_array_fan_in(rand_ctx *rng, float *weights, int width, int height, int stride,
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

static void
randomise_weights_fan_in(RecurNN *net, float sum, float kurtosis,
    float margin, float inputs_weight_ratio){
  memset(net->ih_weights, 0, net->ih_size * sizeof(float));
  memset(net->ho_weights, 0, net->ho_size * sizeof(float));
  int hsize = 1 + net->hidden_size;
  if (inputs_weight_ratio > 0){
    randomise_array_fan_in(&net->rng, net->ih_weights + 1,
        net->hidden_size, hsize, net->h_size, sum, kurtosis, margin);
    randomise_array_fan_in(&net->rng, net->ih_weights + hsize * net->h_size + 1,
        net->hidden_size, net->input_size, net->h_size,
        sum * inputs_weight_ratio, kurtosis, margin);
  }
  else {
    randomise_array_fan_in(&net->rng, net->ih_weights + 1,
        net->hidden_size, hsize + net->input_size, net->h_size,
        sum, kurtosis, margin);
  }
  randomise_array_fan_in(&net->rng, net->ho_weights, net->output_size, net->hidden_size,
      net->o_size, sum, kurtosis, margin);

  if (net->bottom_layer){
    RecurExtraLayer *bl = net->bottom_layer;
    memset(bl->weights, 0, bl->i_size * bl->o_size * sizeof(float));
    randomise_array_fan_in(&net->rng, bl->weights, bl->output_size,
        bl->input_size + 1, bl->o_size,
        sum, kurtosis, margin);
  }
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
    STDERR_DEBUG("used submethod %d%s%s", p->submethod,
        p->bias_uses_submethod ? ", bias too" : "",
        p->inputs_use_submethod ? ", inputs too" : ""
    );
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
    randomise_weights_fan_in(net,
        p->fan_in_sum,
        p->fan_in_step,
        p->fan_in_min,
        p->fan_in_ratio);
  }
  else if (p->method == RNN_INIT_FLAT){
    randomise_weights_flat(net,
        p->flat_variance,
        p->flat_shape,
        p->flat_perforation
    );
  }
  else if (p->method == RNN_INIT_RUNS){
    maybe_randomise_using_submethod(net, p);
    initialise_loops_or_runs(net,
        p->run_n,
        p->run_len_mean,
        p->run_len_stddev,
        p->run_gain,
        p->run_input_probability,
        p->run_input_magnitude,
        p->run_loop,
        p->run_crossing_paths,
        p->run_inputs_miss,
        p->run_input_at_start);
  }
}

void
rnn_init_default_weight_parameters(RecurNN *net,
    struct RecurInitialisationParameters *q){
  struct RecurInitialisationParameters p = {
    /*common to multiple methods */
    .method = RNN_INIT_FLAT,
    .submethod = RNN_INIT_FLAT,
    .bias_uses_submethod = 0,
    .inputs_use_submethod = 0,

    /*used when .method OR .submethod == RNN_INIT_FAN_IN */
    .fan_in_ratio = net->input_size  * 1.0f / net->hidden_size,
    .fan_in_sum = 3.0,
    .fan_in_step = 0.3,
    .fan_in_min = 0.1,

    /*used when .method OR .submethod == RNN_INIT_FLAT */
    .flat_variance = RNN_INITIAL_WEIGHT_VARIANCE_FACTOR / net->h_size,
    .flat_shape = RNN_INIT_DIST_UNIFORM,
    .flat_perforation = 0.7,

    /*RNN_INIT_RUNS */
    .run_input_probability = .17,
    .run_input_magnitude = 0.2,
    .run_gain = 0.17,
    .run_len_mean = net->hidden_size / 1.0,
    .run_len_stddev = net->hidden_size / 3.0f,
    .run_n = net->h_size * 0.085,
    .run_loop = 1,
    .run_crossing_paths = 0,
    .run_inputs_miss = 0,
    .run_input_at_start = 0,
  };
  *q = p;
}

void
rnn_randomise_weights_auto(RecurNN *net){
  /*heuristically choose an initialisation scheme for the size of net */
  /*initial heuristic is very simple*/
  rnn_randomise_weights_simple(net, RNN_INIT_FLAT);
}


void
rnn_randomise_weights_simple(RecurNN *net, const rnn_init_method method){
  struct RecurInitialisationParameters p;
  rnn_init_default_weight_parameters(net, &p);
  p.method = method;
  rnn_randomise_weights_clever(net, &p);
}



void rnn_perforate_weights(RecurNN *net, float p){
  perforate_array(net->ih_weights, net->ih_size, p, &net->rng);
  perforate_array(net->ho_weights, net->ho_size, p, &net->rng);
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

static inline void
print_mean_and_variance(const float *array, int width, int height, int stride,
    int offset, const char *name){
  int x, y;
  float mean = 0;
  float var = 0;
  float n = 0;
  for (y = 0; y < height; y++){
    for (x = offset; x < width + offset; x++){
      n++;
      float val = array[y * stride + x];
      float delta = val - mean;
      mean += delta / n;
      var += delta * (val - mean);
    }
  }
  var /= n;
  STDERR_DEBUG("%s: mean %3g variance %3g (std dev %3g) n %d",
      name, mean, var, sqrt(var), (int)n);
}

void
rnn_print_net_stats(RecurNN *net){
  print_mean_and_variance(net->ih_weights, net->hidden_size,
      net->hidden_size + net->input_size + 1, net->h_size,
      1, "ih_weights");
  print_mean_and_variance(net->ho_weights, net->output_size,
      net->hidden_size + 1, net->o_size,
      0, "ho_weights");

  if (net->bottom_layer){
    RecurExtraLayer *bl = net->bottom_layer;
    print_mean_and_variance(bl->weights, bl->output_size,
        bl->input_size, bl->o_size,
        1, "bottom weights");
  }
}
