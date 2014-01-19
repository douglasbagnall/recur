#include "recur-nn.h"
#include "recur-nn-helpers.h"

static RecurNNBPTT *
new_bptt(RecurNN *net, int depth, float learn_rate, float momentum,
    int batch_size, int use_momentum){
  RecurNNBPTT *bptt = calloc(sizeof(RecurNNBPTT), 1);
  MAYBE_DEBUG("allocated bptt %p", bptt);
  bptt->depth = depth;
  bptt->learn_rate = learn_rate;
  bptt->momentum = momentum;
  bptt->momentum_weight = RNN_MOMENTUM_WEIGHT;
  batch_size = MAX(1, batch_size);
  bptt->batch_size = batch_size;
  size_t vlen = net->i_size * 2 + net->h_size * 0 + net->o_size * 1;
  vlen += net->ih_size + net->ho_size;
  if (use_momentum){
    vlen += net->ih_size + net->ho_size;
  }
  vlen += depth * net->i_size;

  float *fm = zalloc_aligned_or_die(vlen * sizeof(float));
  bptt->mem = fm;
  /*haphazard arrangement of arrays has a point, see comment in rnn_new*/
#define SET_ATTR_SIZE(attr, size) bptt->attr = fm; fm += (size);
  SET_ATTR_SIZE(o_error,           net->o_size);
  if (use_momentum){
    SET_ATTR_SIZE(ih_momentum,       net->ih_size);
    SET_ATTR_SIZE(ho_momentum,       net->ho_size);
  }
  /*h_error uses strictly larger i_size, facilitating switching between the 2*/
  SET_ATTR_SIZE(i_error,           net->i_size);
  SET_ATTR_SIZE(history,           depth * net->i_size);
  SET_ATTR_SIZE(h_error,           net->i_size);
  SET_ATTR_SIZE(ih_delta,          net->ih_size);
  SET_ATTR_SIZE(ho_delta,          net->ho_size);
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
rnn_new(uint input_size, uint hidden_size, uint output_size, int flags,
    u64 rng_seed, const char *log_file, int bptt_depth, float learn_rate,
    float momentum, int batch_size){
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
    net->bptt = new_bptt(net, bptt_depth, learn_rate, momentum, batch_size,
        flags);
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
rnn_clone(RecurNN *parent, int flags,
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
  int batch_size;

  /*XXXX It would be nice to have a half bptt state -- with shared momentum
    but separate deltas, history */
  if (parent->bptt && (flags & RNN_NET_FLAG_OWN_BPTT)){
    learn_rate = parent->bptt->learn_rate;
    bptt_depth = parent->bptt->depth;
    momentum = parent->bptt->momentum;
    batch_size = parent->bptt->batch_size;
  }
  else { /*doesn't matter what these are */
    learn_rate = 0;
    bptt_depth = 0;
    momentum = 0;
    batch_size = 0;
  }

  net = rnn_new(parent->input_size, parent->hidden_size, parent->output_size,
      flags, rng_seed, log_file, bptt_depth, learn_rate, momentum,
      batch_size);

  if (parent->bptt && (flags & RNN_NET_FLAG_OWN_BPTT)){
    net->bptt->momentum_weight = parent->bptt->momentum_weight;
    if (flags & RNN_NET_FLAG_NO_MOMENTUMS){
      net->bptt->ih_momentum = parent->bptt->ih_momentum;
      net->bptt->ho_momentum = parent->bptt->ho_momentum;
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

static inline float
gaussian_power(rand_ctx *rng, float a, int power){
  for (int i = 0; i < power; i++){
    a *= cheap_gaussian_noise(rng);
  }
  return a;
}

/*XXX Glorot and Bengio suggest

  sqrt(6.0 / (inputs + outputs)) * uniform(-1, 1)

  for tanh networks.

  Sutskever recommends sparsely populated weights scaled so that the maximum
  eigenvalue is 1.2.
 */

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
    float margin, int initialise_inputs_separately){
  memset(net->ih_weights, 0, net->ih_size * sizeof(float));
  memset(net->ho_weights, 0, net->ho_size * sizeof(float));

  int hsize = net->bias + net->hidden_size;
  if (initialise_inputs_separately){
    randomise_weights_fan_in(&net->rng, net->ih_weights + net->bias,
        net->hidden_size, hsize, net->h_size, sum, kurtosis, margin);
    randomise_weights_fan_in(&net->rng, net->ih_weights + hsize * net->h_size + net->bias,
        net->hidden_size,
        net->input_size, net->h_size, sum, kurtosis, margin);
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
rnn_forget_history(RecurNN *net, int bptt_too){
  zero_aligned_array(net->hidden_layer, net->h_size);
  /*zero aligned array doesn't work on possibly unaligned length */
  memset(net->input_layer, 0, (net->hidden_size + net->bias) * sizeof(float));
  if (bptt_too && net->bptt){
    zero_aligned_array(net->bptt->history, net->bptt->depth * net->i_size);
  }
}

static inline void
calculate_interlayer(const float *restrict inputs,
    int input_size,
    float *restrict outputs,
    int output_size,
    const float *restrict weights)
{
#if 1
  /* Naive opinion tests ~25% quicker than cblas opinion for scarcely trained
     nets of all sizes, and 50% quicker for highly trained 1999 neuron
     nets.

     Probably the knowledge that the input array is fairly sparse is more
     valuable than atlas/openblas's clever use of cache.
 */
  ASSUME_ALIGNED(inputs);
  ASSUME_ALIGNED(outputs);
  ASSUME_ALIGNED(weights);
  int x, y;
  memset(outputs, 0, output_size * sizeof(float));
  for (y = 0; y < input_size; y++){
    if (inputs[y]){
      const float *row = weights + output_size * y;
      ASSUME_ALIGNED(row);
      for (x = 0; x < output_size; x++){
        outputs[x] += inputs[y] * row[x];
      }
    }
  }
#else
  /* cblas */
  cblas_sgemv(
      CblasRowMajor, /*CblasColMajor, CblasRowMajor*/
      CblasTrans,   /*CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113*/
      input_size,
      output_size,
      1.0,
      weights,
      output_size,  /*"LDA", stride*/
      inputs,
      1,
      0,
      outputs,
      1);
#endif
}


static inline void
maybe_scale_hiddens(RecurNN *net){
  float *hiddens = net->hidden_layer;
  ASSUME_ALIGNED(hiddens);
  float softclip = net->h_size * HIDDEN_MEAN_SOFT_TOP;
  float sum = 0.0f;
  for (int i = 0; i < net->h_size; i++){
    sum += hiddens[i];
  }
  if (sum > softclip){
    softclip_scale(sum, softclip, hiddens, net->h_size);
#if 0
    /*scale the weights as well, but not quite as much */
    float scale = (1.0f + soft_clip(sum, softclip)) * 0.5f;
    scale_aligned_array(net->ih_weights, net->ih_size, scale);
    if (net->bptt && net->bptt->ih_momentum){
      scale_aligned_array(net->bptt->ih_momentum, net->ih_size, scale);
    }
    MAYBE_DEBUG("scaling weights (hidden sum %f > %f)", sum, softclip);
    rnn_log_float(net, "weight_scale", scale);
#endif
  }
}

static inline void
maybe_scale_inputs(RecurNN *net){
  float *inputs = net->input_layer;
  ASSUME_ALIGNED(inputs);
  float softclip = net->i_size * INPUT_MEAN_SOFT_TOP;
  float sum = 0.0f;
  for (int i = 0; i < net->i_size; i++){
    sum += inputs[i];
  }
  if (sum > softclip){
    softclip_scale(sum, softclip, inputs, net->i_size);
    MAYBE_DEBUG("scaling inputs (input sum %f > %f)", sum, softclip);
  }
}



/*raw opinion does the core calculation */
static void
rnn_raw_opinion(RecurNN *net){
  if (net->bias)
    net->input_layer[0] = 1.0f;

  maybe_scale_inputs(net);

  calculate_interlayer(net->input_layer, net->i_size,
      net->hidden_layer, net->h_size, net->ih_weights);

  ASSUME_ALIGNED(net->hidden_layer);
  for (int i = 0; i < net->h_size; i++){
    net->hidden_layer[i] -= RNN_HIDDEN_PENALTY;
    net->hidden_layer[i] = MAX(net->hidden_layer[i], 0.0);
  }
  maybe_scale_hiddens(net);

  if (net->bias)
    net->hidden_layer[0] = 1.0f;
  for (int i = net->hidden_size + net->bias; i < net->h_size; i++){
    net->hidden_layer[i] = 0;
  }
  calculate_interlayer(net->hidden_layer, net->h_size,
      net->output_layer, net->o_size, net->ho_weights);
}

float *
rnn_opinion(RecurNN *net, const float *inputs){
  if (inputs){
    memcpy(net->real_inputs, inputs, net->input_size * sizeof(float));
  }
  /*copy in hiddens */
  int hsize = net->hidden_size + net->bias;
  memcpy(net->input_layer, net->hidden_layer, hsize * sizeof(float));
  rnn_raw_opinion(net);
  return net->output_layer;
}

float *
rnn_opinion_with_dropout(RecurNN *net, const float *inputs, float dropout){
  int hsize = net->hidden_size + net->bias;
  int isize = net->input_size;
  if (inputs){
    memcpy(net->real_inputs, inputs, isize * sizeof(float));
  }
  /*copy in hiddens */
  memcpy(net->input_layer, net->hidden_layer, hsize * sizeof(float));

  if (net->bias){
    net->input_layer[0] = 1.0f;
  }
  maybe_scale_inputs(net);
  dropout_array(net->input_layer, hsize, dropout, &net->rng);

  calculate_interlayer(net->input_layer, net->i_size,
      net->hidden_layer, net->h_size, net->ih_weights);

  float s = hsize + isize;
  float dropout_scale = s / (s - hsize * dropout);

  ASSUME_ALIGNED(net->hidden_layer);
  for (int i = 0; i < net->h_size; i++){
    net->hidden_layer[i] -= RNN_HIDDEN_PENALTY;
    if (net->hidden_layer[i] > 0){
      net->hidden_layer[i] *= dropout_scale;
    }
    else{
      net->hidden_layer[i] = 0.0f;
    }
  }
  maybe_scale_hiddens(net);

  if (net->bias)
    net->hidden_layer[0] = 1.0f;
  for (int i = net->hidden_size + net->bias; i < net->h_size; i++){
    net->hidden_layer[i] = 0;
  }
  calculate_interlayer(net->hidden_layer, net->h_size,
      net->output_layer, net->o_size, net->ho_weights);

  return net->output_layer;
}


static inline float
backprop_top_layer(RecurNN *net)
{
  int x, y;
  float error_sum = 0.0f;

  const float *restrict hiddens = net->hidden_layer;
  float *restrict h_error = net->bptt->h_error;
  const float *restrict o_error = net->bptt->o_error;
  const float *restrict weights = net->ho_weights;
  ASSUME_ALIGNED(hiddens);
  ASSUME_ALIGNED(h_error);
  ASSUME_ALIGNED(o_error);
  ASSUME_ALIGNED(weights);

  for (y = net->bias; y < net->h_size; y++){
    float e = 0.0f;
    if (hiddens[y]){
      const float *restrict row = weights + y * net->o_size;
      ASSUME_ALIGNED(row);
      for (x = 0; x < net->o_size; x++){
        e += row[x] * o_error[x];
      }
      error_sum += fabsf(e);
    }
    h_error[y] = e;
  }
  return error_sum;
}

/*apply_sgd_top_layer backpropagates error, calculates updates via gradient
  descent, and alters the weights accordingly.

It is more efficient than calc_sgd_top_layer (with subsequent weight
adjustment) when the top layer synchronic batch size is one.
*/

static float
apply_sgd_top_layer(RecurNN *net){
  //cblas_ger
  RecurNNBPTT *bptt = net->bptt;
  const float *restrict o_error = bptt->o_error;
  float *restrict hiddens = net->hidden_layer;
  float *restrict weights = net->ho_weights;
  float *restrict momentums = bptt->ho_momentum;
  float rate = bptt->learn_rate;
  float momentum = bptt->momentum;
  float momentum_weight = bptt->momentum_weight;
  float error_sum;
  ASSUME_ALIGNED(hiddens);
  ASSUME_ALIGNED(o_error);
  ASSUME_ALIGNED(weights);
  ASSUME_ALIGNED(momentums);

  int y, x;

  if (net->bias){
    hiddens[0] = 1.0f;
  }

  error_sum = backprop_top_layer(net);

  for (y = 0; y < net->h_size; y++){
    float *restrict momentum_row = momentums + y * net->o_size;
    float *restrict row = weights + y * net->o_size;
    ASSUME_ALIGNED(row);
    ASSUME_ALIGNED(momentum_row);

    if (hiddens[y]){
      float m = hiddens[y] * rate;
      for (x = 0; x < net->o_size; x++){
        float d = o_error[x] * m;
        float mm = momentum_row[x];
        row[x] += d + mm * momentum_weight;
        mm += d;
        momentum_row[x] = mm * momentum;
      }
    }
    else {
      for (x = 0; x < net->o_size; x++){
        float mm = momentum_row[x];
        row[x] += mm * momentum_weight;
        momentum_row[x] = mm * momentum;
      }
    }
  }
  return error_sum;
}

/*calc_sgd_top_layer backpropagates error, and calculates weight updates via
  gradient descent, which get put in the net->bptt->ho_delta array.
*/

static float
calc_sgd_top_layer(RecurNN *net){
  //cblas_ger
  RecurNNBPTT *bptt = net->bptt;
  const float *restrict o_error = bptt->o_error;
  float *restrict hiddens = net->hidden_layer;
  float *restrict delta = bptt->ho_delta;
  float error_sum = 0;
  ASSUME_ALIGNED(hiddens);
  ASSUME_ALIGNED(o_error);
  ASSUME_ALIGNED(delta);

  int y, x;

  if (net->bias){
    hiddens[0] = 1.0f;
  }
  MAYBE_DEBUG("top error:[0] %g, [1] %g", o_error[0], o_error[1]);
  error_sum = backprop_top_layer(net);
  zero_aligned_array(delta, net->ho_size);
  for (y = 0; y < net->h_size; y++){
    if (hiddens[y]){
      float *restrict drow = delta + y * net->o_size;
      ASSUME_ALIGNED(drow);
      for (x = 0; x < net->o_size; x++){
        drow[x] = o_error[x] * hiddens[y];
      }
    }
    MAYBE_DEBUG("y: %d drow[0] %g [1] %g", y,
        *(delta + y * net->o_size),
        *(delta + y * net->o_size + 1));
  }

  return error_sum;
}



static float
bptt_and_accumulate_error(RecurNN *net, float *ih_delta, const float top_error_sum)
{
  RecurNNBPTT *bptt = net->bptt;
  int y, x;
  float *restrict h_error = bptt->h_error;
  float *restrict i_error = bptt->i_error;
  const float *restrict weights = net->ih_weights;
  ASSUME_ALIGNED(weights);
  ASSUME_ALIGNED(ih_delta);
  ASSUME_ALIGNED(h_error);
  ASSUME_ALIGNED(i_error);
#if VECTOR
  int vhsize = net->h_size / 4;
#endif

  float error_sum = 0;
  float max_error_sum = MAX_ERROR_GAIN * top_error_sum;
  float min_error_gain = MIN_ERROR_GAIN * top_error_sum;
  float min_error_sum = MIN(bptt->min_error_factor / net->bptt->learn_rate,
      min_error_gain);
  int t;
  for (t = bptt->depth; t > 0; t--){
    error_sum = 0.0f;
    int offset = (t + bptt->index) % bptt->depth;
    const float *restrict inputs = bptt->history + offset * net->i_size;
    ASSUME_ALIGNED(inputs);
    if (net->bias){
      h_error[0] = 0.0;
    }
    for (int i = net->hidden_size + net->bias; i < net->h_size; i++){
      h_error[i] = 0.0;
    }
    for (y = 0; y < net->i_size; y++){
      if (inputs[y] != 0.0f){
        float e;
        const float *restrict w_row = weights + y * net->h_size;
        float *restrict delta_row = ih_delta + y * net->h_size;
        ASSUME_ALIGNED(w_row);
        ASSUME_ALIGNED(delta_row);
#if VECTOR
        v4ss *restrict vh_error = (v4ss*)h_error;
        v4ss ve = {0, 0, 0, 0};
        v4ss inv = {inputs[y], inputs[y], inputs[y], inputs[y]};
        v4ss *restrict vd = (v4ss*)delta_row;
        v4ss *restrict vw = (v4ss*)w_row;
        for (x = 0; x < vhsize; x++){
          vd[x] += vh_error[x] * inv;
          ve += vw[x] * vh_error[x];
        }
        e = ve[0] + ve[1] + ve[2] + ve[3];
#else
        e = 0.0f;
        for (x = 0; x < net->h_size; x++){
          delta_row[x] += h_error[x] * inputs[y];
          e += w_row[x] * h_error[x];
        }
#endif
        i_error[y] = e;
        error_sum += e * e;
      }
      else {
        i_error[y] = 0;
      }
    }
    float *tmp = h_error;
    h_error = i_error;
    i_error = tmp;
    if (error_sum < min_error_sum || error_sum > max_error_sum){
      break;
    }
  }

  if (error_sum > max_error_sum){
    bptt->ih_scale = soft_clip(error_sum, max_error_sum);
  }
  else {
    bptt->ih_scale = 1.0f;
    if (net->flags & RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR){
      int depth_error = bptt->depth / 4 - t;
      if (bptt->min_error_factor < MAX_MIN_ERROR_FACTOR &&
          (min_error_gain != min_error_sum || depth_error < 0)){
        bptt->min_error_factor *= (1.0f + depth_error * 1e-3);
      }
      bptt->min_error_factor = MAX(bptt->min_error_factor, ABS_MIN_ERROR_FACTOR);
    }
  }

  if (net->log){
    rnn_log_int(net, "depth", bptt->depth - t);
    rnn_log_float(net, "scaled_error", bptt->ih_scale * error_sum);
    rnn_log_float(net, "ih_scale", bptt->ih_scale);
    rnn_log_float(net, "min_error_threshold", min_error_sum);
    rnn_log_float(net, "min_error_factor", bptt->min_error_factor);
    if (net->flags & RNN_NET_FLAG_LOG_HIDDEN_SUM){
      float hidden_sum = 0;
      for (int i = 0; i < net->h_size; i++){
        hidden_sum += net->hidden_layer[i];
      }
      rnn_log_float(net, "hidden_sum", hidden_sum);
    }
    if (net->flags & RNN_NET_FLAG_LOG_WEIGHT_SUM){
      float weight_sum = abs_sum_aligned_array(weights, net->ih_size);
      rnn_log_float(net, "weight_sum", weight_sum);
    }
  }
  return error_sum;
}

/*apply_learning_with_momentum updates weights and momentum according to
  delta and momentum. */
static void
apply_learning_with_momentum(float *restrict weights,
    const float *restrict delta, float *restrict momentums,
    int size, const float rate, const float momentum, const float momentum_weight){

  ASSUME_ALIGNED(weights);
  ASSUME_ALIGNED(delta);
  ASSUME_ALIGNED(momentums);

/*GCC actually does as well or better with its own vectorisation*/
#if VECTOR_ALL_THE_WAY

  size /= 4;
  v4ss rate_v = {rate, rate, rate, rate};
  v4ss momentum_v = {momentum, momentum, momentum, momentum};
  v4ss momentum_weight_v = {momentum_weight, momentum_weight,
                            momentum_weight, momentum_weight};
  v4ss *vd = (v4ss*)delta;
  v4ss *vw = (v4ss*)weights;
  v4ss *vm = (v4ss*)momentums;
  for (int i = 0; i < size; i++){
    v4ss t = vd[i] * rate_v;
    v4ss m = vm[i];
    vm[i] = (m + t) * momentum_v;
    vw[i] += t + m * momentum_weight_v;
  }

#else
  for (int i = 0; i < size; i++){
    float t = delta[i] * rate;
    float m = momentums[i];
    weights[i] += t + m * momentum_weight;
    momentums[i] = (m + t) * momentum;
  }
#endif
  MAYBE_DEBUG("momentums %.2g %.2g %.2g %.2g %.2g",
      momentums[0], momentums[3], momentums[6], momentums[10], momentums[11]);
}


void
rnn_prepare_nesterov_momentum(RecurNN *net){
  scale_aligned_array(net->bptt->ih_momentum, net->ih_size, net->bptt->momentum);
  add_aligned_arrays(net->ih_weights, net->ih_size, net->bptt->ih_momentum, 1.0f);
  scale_aligned_array(net->bptt->ho_momentum, net->ho_size, net->bptt->momentum);
  add_aligned_arrays(net->ho_weights, net->ho_size, net->bptt->ho_momentum, 1.0f);
}

/*with standard Nesterov momentum, the momentum has previously been scaled and
  added to the weights.*/
static void
apply_learning_with_nesterov_momentum(float *restrict momentums,
    const float *restrict delta, float *restrict weights,
    int size, const float rate){
  ASSUME_ALIGNED(momentums);
  ASSUME_ALIGNED(delta);
  ASSUME_ALIGNED(weights);
  for (int i = 0; i < size; i++){
    float t = delta[i] * rate;
    weights[i] += t;
    momentums[i] += t;
  }
}


static inline float
apply_sgd_with_bptt(RecurNN *net, float top_error_sum){
  RecurNNBPTT *bptt = net->bptt;
  zero_aligned_array(bptt->ih_delta, net->ih_size);
  float error_sum = bptt_and_accumulate_error(net, bptt->ih_delta, top_error_sum);
  float rate = bptt->learn_rate * bptt->ih_scale;

  apply_learning_with_momentum(net->ih_weights, bptt->ih_delta, bptt->ih_momentum,
      net->ih_size, rate, bptt->momentum, bptt->momentum_weight);
  return error_sum;
}

static inline float
apply_sgd_with_bptt_batch(RecurNN *net, float top_error_sum){
  RecurNNBPTT *bptt = net->bptt;
  float rate = bptt->learn_rate;
  float *gradient = zalloc_aligned_or_die(net->ih_size * sizeof(float));
  float error_sum = bptt_and_accumulate_error(net, gradient, top_error_sum);

  add_aligned_arrays(bptt->ih_delta, net->ih_size, gradient, bptt->ih_scale);

  free(gradient);

  if ((net->generation % bptt->batch_size) == 0){
    apply_learning_with_momentum(net->ih_weights, bptt->ih_delta, bptt->ih_momentum,
        net->ih_size, rate, bptt->momentum, bptt->momentum_weight);
    zero_aligned_array(bptt->ih_delta, net->ih_size);
  }
  return error_sum;
}

void rnn_consolidate_many_nets(RecurNN **nets, int n, int momentum_style,
    float momentum_soft_start){
  RecurNN *net = nets[0];
  RecurNNBPTT *bptt = net->bptt;
  /*Use first net's delta as gradient accumulator.*/
  float *ho_gradient = bptt->ho_delta;
  float *ih_gradient = bptt->ih_delta;
  int ho_size = net->ho_size;
  int ih_size = net->ih_size;

  float momentum = bptt->momentum;
  if (momentum_soft_start){
    /*XXX keeps doing the calculation long after soft_start is over. */
    float x = momentum_soft_start;
    momentum = MIN(momentum, 1.0f - x / (1 + net->generation + 2 * x));
  }
  float momentum_weight;

  switch (momentum_style){
  case RNN_MOMENTUM_SIMPLIFIED_NESTEROV:
    momentum_weight = momentum / (1.0 + momentum);
    break;
  case RNN_MOMENTUM_CLASSICAL:
    momentum_weight = 1.0f;
    break;
  default:
    momentum_weight = bptt->momentum_weight;
    break;
  }

  for (int i = 1; i < n; i++){
    add_aligned_arrays(ho_gradient, ho_size, nets[i]->bptt->ho_delta, 1.0f);
  }
  switch (momentum_style){
  case RNN_MOMENTUM_NESTEROV:
    apply_learning_with_nesterov_momentum(bptt->ho_momentum, ho_gradient,
        net->ho_weights, net->ho_size, bptt->learn_rate * bptt->ho_scale);
    break;
  default:
    apply_learning_with_momentum(net->ho_weights, ho_gradient, bptt->ho_momentum,
        net->ho_size, bptt->learn_rate * bptt->ho_scale,
        momentum, momentum_weight);
    break;
  }

  /*the various gradients may to be scaled before summing*/
  if (bptt->ih_scale != 1.0f){
    scale_aligned_array(ih_gradient, net->ih_size, bptt->ih_scale);
  }
  for (int i = 1; i < n; i++){
    add_aligned_arrays(ih_gradient, ih_size, nets[i]->bptt->ih_delta,
        nets[i]->bptt->ih_scale);
  }

  switch (momentum_style){
  case RNN_MOMENTUM_NESTEROV:
    apply_learning_with_nesterov_momentum(bptt->ih_momentum, ih_gradient,
        net->ih_weights, net->ih_size, bptt->learn_rate);
    break;
  default:
    apply_learning_with_momentum(net->ih_weights, ih_gradient, bptt->ih_momentum,
        net->ih_size, bptt->learn_rate, momentum, momentum_weight);
    break;
  }
  rnn_log_float(net, "momentum", momentum);
  rnn_log_float(net, "momentum_weight", momentum_weight);
}


void
rnn_bptt_advance(RecurNN *net){
  RecurNNBPTT *bptt = net->bptt;
  bptt->index++;
  if (bptt->index == bptt->depth)
    bptt->index -= bptt->depth;
  net->input_layer = bptt->history + bptt->index * net->i_size;
  net->real_inputs = net->input_layer + net->bias + net->hidden_size;
}

void
rnn_bptt_calc_deltas(RecurNN *net){
  float top_error_sum = calc_sgd_top_layer(net);
  float top_error_scaled = softclip_scale(top_error_sum,
      net->h_size * MAX_TOP_ERROR_FACTOR, net->bptt->h_error, net->h_size);

  zero_aligned_array(net->bptt->ih_delta, net->ih_size);
  float bptt_error_sum = bptt_and_accumulate_error(net,
      net->bptt->ih_delta, top_error_scaled);
  net->generation++;
  if (net->log){
    rnn_log_float(net, "error_gain", bptt_error_sum / (top_error_scaled + 1e-6));
    rnn_log_float(net, "top_error_scaled", top_error_scaled);
    rnn_log_float(net, "top_error_raw", top_error_sum);
    rnn_log_int(net, "generation", net->generation);
  }
}

void
rnn_bptt_calculate(RecurNN *net){
  float bptt_error_sum;
  float top_error_sum = apply_sgd_top_layer(net);
  float top_error_scaled = softclip_scale(top_error_sum,
      net->h_size * MAX_TOP_ERROR_FACTOR, net->bptt->h_error, net->h_size);

  if (net->bptt->batch_size > 1)
    bptt_error_sum = apply_sgd_with_bptt_batch(net, top_error_scaled);
  else
    bptt_error_sum = apply_sgd_with_bptt(net, top_error_scaled);
  net->generation++;
  if (net->log){
    rnn_log_float(net, "top_error_scaled", top_error_scaled);
    rnn_log_float(net, "top_error_raw", top_error_sum);
    rnn_log_float(net, "error_sum", bptt_error_sum);
    rnn_log_float(net, "error_gain", bptt_error_sum / (top_error_scaled + 1e-6));
    rnn_log_int(net, "generation", net->generation);
  }
  rnn_condition_net(net);
}

/*rnn_condition_nets performs various periodic operations to keep the numbers
  in good order (not too big, not too small). The different operations occur
  at different points in the periodic cycle, so as not to freeze.

  Certain bits in net->flags mask off the various operations.
*/

void
rnn_condition_net(RecurNN *net)
{
  u32 mask = net->flags >> RNN_COND_USE_OFFSET;
  u32 m = net->generation % RNN_CONDITIONING_INTERVAL;
  MAYBE_DEBUG("flags %x mask  %x m %x, hit %x",
      net->flags, mask, m, (1 << m) & mask);


  if (((1 << m) & mask) == 0){
    return;
  }
  switch (m){
  case RNN_COND_BIT_SCALE:
    /*XXX not scaling momentums.*/
    scale_aligned_array(net->ih_weights, net->ih_size, WEIGHT_SCALE);
    scale_aligned_array(net->ho_weights, net->ho_size, WEIGHT_SCALE);
    break;
  case RNN_COND_BIT_ZERO:
    zero_small_numbers(net->ih_weights, net->ih_size);
    zero_small_numbers(net->ho_weights, net->ho_size);
    if (net->bptt){
      zero_small_numbers(net->bptt->ih_momentum, net->ih_size);
      zero_small_numbers(net->bptt->ho_momentum, net->ho_size);
    }
    break;
  case RNN_COND_BIT_RAND:
    {
      int t = rand_small_int(&net->rng, net->ih_size + net->ho_size);
      float damage = (cheap_gaussian_noise(&net->rng) *
          RANDOM_DAMAGE_FACTOR * net->h_size * net->bptt->learn_rate);
      if (t >= net->ih_size){
        t -= net->ih_size;
        int col = t % net->o_size;
        if (col < net->output_size){
          net->ho_weights[t] += damage;
        }
      }
      else {
        int col = t % net->h_size;
        if (col >= net->bias && col < net->hidden_size + net->bias){
          net->ih_weights[t] += damage;
        }
      }
    }
    break;
  case RNN_COND_BIT_TALL_POPPY:
    {
      int big_i = 0;
      float big_v = fabsf(net->ih_weights[0]);
      for (int i = 1; i < net->ih_size; i++){
        float v = fabsf(net->ih_weights[i]);
        if (v > big_v){
          big_v = v;
          big_i = i;
        }
      }
      if (big_v > RNN_TALL_POPPY_THRESHOLD){
        net->ih_weights[big_i] *= RNN_TALL_POPPY_SCALE;
      }
      MAYBE_DEBUG("reducing weight %d from %.2g to %.2g",
          big_i, big_v, net->ih_weights[big_i]);
    }
    break;
  case RNN_COND_BIT_LAWN_MOWER:
    {
      for (int i = 0; i < net->ih_size; i++){
        net->ih_weights[i] = MAX(net->ih_weights[i], -RNN_LAWN_MOWER_THRESHOLD);
        net->ih_weights[i] = MIN(net->ih_weights[i],  RNN_LAWN_MOWER_THRESHOLD);
      }
    }
    break;
  }
}

/*duplicates calculations already done elsewhere, but when the net is used in
  parallel, it would be messy to log every use. */
void rnn_log_net(RecurNN *net)
{
  int i;
  if (net->log == NULL)
    return;
  float top_error = 0;
  float hidden_error = 0;
  if (net->bptt){
    for (i = 0; i < net->o_size; i++){
      top_error += fabsf(net->bptt->o_error[i]);
    }
    for (i = 0; i < net->h_size; i++){
      hidden_error += fabsf(net->bptt->h_error[i]);
    }
    rnn_log_float(net, "output_error", top_error);
    rnn_log_float(net, "hidden_error", hidden_error);
  }
#if 0
  float hidden_sum = 0;
  for (i = 0; i < net->h_size; i++){
    hidden_sum += net->hidden_layer[i];
  }
  rnn_log_float(net, "hidden_sum", hidden_sum);
#endif
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
