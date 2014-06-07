/* Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL/MPL2 */
#include "recur-nn.h"
#include "recur-nn-helpers.h"

#define INPUT_OFFSET(net) ((net)->hidden_size + 1)

void
rnn_forget_history(RecurNN *net, int bptt_too){
  zero_aligned_array(net->hidden_layer, net->h_size);
  /*zero aligned array doesn't work on possibly unaligned length */
  memset(net->input_layer, 0, INPUT_OFFSET(net) * sizeof(float));
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
#if ! USE_CBLAS
  /* Naive opinion tests ~25% quicker than cblas opinion for scarcely trained
     nets of all sizes, and 50% quicker for highly trained 1999 neuron
     nets.

     Probably the knowledge that the input array is fairly sparse is more
     valuable than atlas/openblas's clever use of cache.
 */
  ASSUME_ALIGNED(inputs);
  ASSUME_ALIGNED(outputs);
  ASSUME_ALIGNED(weights);
  ASSUME_ALIGNED_LENGTH(output_size);
  int x, y;
  zero_aligned_array(outputs, output_size);
  for (y = 0; y < input_size; y++){
    float input = inputs[y];
    if (input){
      const float *row = weights + output_size * y;
      ASSUME_ALIGNED(row);
      for (x = 0; x < output_size; x++){
        outputs[x] += input * row[x];
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

float *
rnn_opinion(RecurNN *net, const float *restrict inputs){
  /*If inputs is NULL, assume the inputs have already been set.*/
  float *restrict hiddens = net->hidden_layer;
  ASSUME_ALIGNED(hiddens);
  if (net->bottom_layer){
    /*possible bottom layer */
    RecurExtraLayer *layer = net->bottom_layer;
    layer->inputs[0] = 1.0f; /*bias*/
    if (inputs){
      memcpy(layer->inputs + 1, inputs, layer->input_size * sizeof(float));
    }
    calculate_interlayer(layer->inputs, layer->i_size, layer->outputs,
         layer->o_size, layer->weights);
    memcpy(net->real_inputs, layer->outputs, net->input_size * sizeof(float));
  }
  else if (inputs){
    memcpy(net->real_inputs, inputs, net->input_size * sizeof(float));
  }

  /*copy in hiddens */
  memcpy(net->input_layer, hiddens, INPUT_OFFSET(net) * sizeof(float));

  /*bias, possibly unnecessary because it may not get overwritten */
  net->input_layer[0] = 1.0f;

  /* in emergencies, clamp the scale of the input vector */
  maybe_scale_inputs(net);

  calculate_interlayer(net->input_layer, net->i_size,
      hiddens, net->h_size, net->ih_weights);
  for (int i = 1; i < net->h_size; i++){
    float h = hiddens[i] - RNN_HIDDEN_PENALTY;
    hiddens[i] = (h > 0.0f) ? h : 0.0f;
  }

  hiddens[0] = 1.0f;

  calculate_interlayer(hiddens, net->h_size,
      net->output_layer, net->o_size, net->ho_weights);

  return net->output_layer;
}

static inline float
backprop_single_layer(
    const float *restrict weights,
    const float *restrict inputs,
    float *restrict i_error,
    int i_size,
    const float *restrict o_error,
    int o_size)
{
  int x, y;
  float error_sum = 0.0f;
  ASSUME_ALIGNED(inputs);
  ASSUME_ALIGNED(i_error);
  ASSUME_ALIGNED(o_error);
  ASSUME_ALIGNED(weights);

  for (y = 1; y < i_size; y++){
    float e = 0.0f;
    if (inputs[y]){
      const float *restrict row = weights + y * o_size;
      ASSUME_ALIGNED(row);
      for (x = 0; x < o_size; x++){
        e += row[x] * o_error[x];
      }
      error_sum += fabsf(e);
    }
    i_error[y] = e;
  }
  return error_sum;
}



static inline float
backprop_top_layer(RecurNN *net)
{
  return backprop_single_layer(
      net->ho_weights,
      net->hidden_layer,
      net->bptt->h_error,
      net->h_size,
      net->bptt->o_error,
      net->o_size);
}

/*single_layer_sgd does gradient descent for a single layer (i.e., top layer
  or extra bottom layers), but it DOESN'T zero the delta array first. That
  is, it will accumulate gradient over several runs. */
static inline void
single_layer_sgd(float const *restrict inputs, int i_size, const float *restrict o_error,
    int o_size, float *restrict deltas){
  ASSUME_ALIGNED(inputs);
  ASSUME_ALIGNED(o_error);
  ASSUME_ALIGNED(deltas);
  int x, y;
  for (y = 0; y < i_size; y++){
    float input = inputs[y];
    if (input){
      float *restrict drow = deltas + y * o_size;
      ASSUME_ALIGNED(drow);
      for (x = 0; x < o_size; x++){
        drow[x] += o_error[x] * input;
      }
    }
  }
}

static float
bptt_and_accumulate_error(RecurNN *net, float *restrict ih_delta,
    float *restrict cumulative_input_error, const float top_error_sum)
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

  float error_sum = 0;
  float max_error_sum = MAX_ERROR_GAIN * top_error_sum;
  float error_sum_ceiling = ERROR_GAIN_CEILING * top_error_sum;
  float min_error_gain = MIN_ERROR_GAIN * top_error_sum;
  float min_error_sum = MIN(bptt->min_error_factor / net->bptt->learn_rate,
      min_error_gain);
  int t;

#if VECTOR
  int vhsize = net->h_size / 4;
#endif
  int offset = bptt->index;
  float cum_error = 0.0f;
  for (t = bptt->depth; t > 0; t--, offset += offset ? -1 : bptt->depth - 1){
    error_sum = 0.0f;
    const float *restrict inputs = bptt->history + offset * net->i_size;
    ASSUME_ALIGNED(inputs);
    h_error[0] = 0.0;
    for (int i = INPUT_OFFSET(net); i < net->h_size; i++){
      h_error[i] = 0.0;
    }
    for (y = 0; y < net->i_size; y++){
      float input = inputs[y];
      if (input != 0.0f){
        float e;
        const float *restrict w_row = weights + y * net->h_size;
        float *restrict delta_row = ih_delta + y * net->h_size;
        ASSUME_ALIGNED(w_row);
        ASSUME_ALIGNED(delta_row);
#if VECTOR
        v4ss *restrict vh_error = (v4ss*)h_error;
        v4ss ve = {0, 0, 0, 0};
        v4ss inv = {input, input, input, input};
        v4ss *restrict vd = (v4ss*)delta_row;
        v4ss *restrict vw = (v4ss*)w_row;
        for (x = 0; x < vhsize; x++){
          v4ss vex = vh_error[x];
          vd[x] += vex * inv;
          ve += vw[x] * vex;
        }
        e = ve[0] + ve[1] + ve[2] + ve[3];
#else
        e = 0.0f;
        for (x = 0; x < net->h_size; x++){
          float ex = h_error[x];
          delta_row[x] += ex * input;
          e += w_row[x] * ex;
        }
#endif
        i_error[y] = e;
        error_sum += e * e;
      }
      else {
        i_error[y] = 0;
      }
    }
    if (cumulative_input_error){
      float *input_error = i_error + INPUT_OFFSET(net);
      for (y = 0; y < net->input_size; y++){
        cumulative_input_error[y] += input_error[y];
      }
    }
    cum_error += sqrtf(error_sum);
    float *tmp = h_error;
    h_error = i_error;
    i_error = tmp;
    if (error_sum <= min_error_sum || error_sum > max_error_sum){
      break;
    }
  }


  if (error_sum > error_sum_ceiling){
    bptt->ih_scale = soft_clip(error_sum, max_error_sum);
    if (cumulative_input_error){
      for (y = 0; y < net->input_size; y++){
        /*doubly shrink cumulative_input_error, to preserve stability in the
          input features */
        cumulative_input_error[y] *= bptt->ih_scale * bptt->ih_scale;
      }
    }
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
    rnn_log_float(net, "cum_error", cum_error);
    if (cumulative_input_error){
      float cie = 0;
      for (y = 0; y < net->input_size; y++){
        cie += cumulative_input_error[y];
      }
      rnn_log_float(net, "cum_input_error", cie);
    }
    if (net->flags & RNN_NET_FLAG_LOG_HIDDEN_SUM){
      float hidden_sum = 0;
      int hidden_zeros = 0;
      float hidden_magnitude = 0;
      float *restrict hiddens = net->hidden_layer;
      for (int i = 0; i < net->h_size; i++){
        float h = hiddens[i];
        hidden_sum += h;
        hidden_magnitude += h * h;
        hidden_zeros += (h == 0.0f);
      }
      rnn_log_float(net, "hidden_sum", hidden_sum);
      rnn_log_float(net, "hidden_magnitude", sqrtf(hidden_magnitude));
      rnn_log_float(net, "hidden_zeros", hidden_zeros / (float)net->hidden_size);
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
}

/*with standard Nesterov momentum, the momentum has previously been scaled and
  added to the weights. Here we do it in the reverse order, so the scaling and
  adding happens at the end in preparation for the next round.*/
static void
apply_learning_with_nesterov_momentum(float *restrict weights,
    const float *restrict delta, float *restrict momentums,
    int size, const float rate, const float momentum){
  ASSUME_ALIGNED(momentums);
  ASSUME_ALIGNED(delta);
  ASSUME_ALIGNED(weights);
  for (int i = 0; i < size; i++){
    float t = delta[i] * rate;
    weights[i] += t;
    momentums[i] += t;
  }

  scale_aligned_array(momentums, size, momentum);
  add_aligned_arrays(weights, size, momentums, 1.0f);
}

float
rnn_calculate_momentum_soft_start(float generation, float max_momentum, float x)
{
  return MIN(max_momentum, 1.0f - x / (1.0f + generation + 2.0f * x));
}

void
rnn_apply_learning(RecurNN *net, int momentum_style,
    float momentum){
  RecurNNBPTT *bptt = net->bptt;
  RecurExtraLayer *bl = net->bottom_layer;
  if (momentum_style == RNN_MOMENTUM_NESTEROV){
    apply_learning_with_nesterov_momentum(net->ho_weights, bptt->ho_delta,
        bptt->ho_momentum, net->ho_size, bptt->learn_rate * bptt->ho_scale, momentum);
    apply_learning_with_nesterov_momentum(net->ih_weights, bptt->ih_delta,
        bptt->ih_momentum, net->ih_size, bptt->learn_rate, momentum);
    if (bl){
      apply_learning_with_nesterov_momentum(bl->weights, bl->delta, bl->momentums,
          bl->i_size * bl->o_size, net->bptt->learn_rate * bl->learn_rate_scale,
          momentum);
    }
  }
  else {
    float momentum_weight;
    if (momentum_style == RNN_MOMENTUM_SIMPLIFIED_NESTEROV){
      momentum_weight = momentum / (1.0 + momentum);
    }
    else if (momentum_style == RNN_MOMENTUM_CLASSICAL){
      momentum_weight = 1.0f;
    }
    else {
      momentum_weight = bptt->momentum_weight;
    }

    apply_learning_with_momentum(net->ho_weights, bptt->ho_delta,
        bptt->ho_momentum, net->ho_size, bptt->learn_rate * bptt->ho_scale,
        momentum, momentum_weight);

    apply_learning_with_momentum(net->ih_weights, bptt->ih_delta,
        bptt->ih_momentum, net->ih_size, bptt->learn_rate, momentum, momentum_weight);

    if (bl){
      apply_learning_with_momentum(bl->weights, bl->delta, bl->momentums,
          bl->i_size * bl->o_size, net->bptt->learn_rate * bl->learn_rate_scale,
          momentum, momentum_weight);
    }
  }
  rnn_log_float(net, "momentum", momentum);
}


void
rnn_bptt_clear_deltas(RecurNN *net)
{
  RecurNNBPTT *bptt = net->bptt;
  zero_aligned_array(bptt->ih_delta, net->ih_size);
  zero_aligned_array(bptt->ho_delta, net->ho_size);
  if (net->bottom_layer){
    zero_aligned_array(net->bottom_layer->o_error,
        net->bottom_layer->o_size);
    zero_aligned_array(net->bottom_layer->delta,
        net->bottom_layer->i_size * net->bottom_layer->o_size);
  }
}


void
rnn_bptt_advance(RecurNN *net){
  RecurNNBPTT *bptt = net->bptt;
  bptt->index++;
  if (bptt->index == bptt->depth)
    bptt->index -= bptt->depth;
  net->input_layer = bptt->history + bptt->index * net->i_size;
  net->real_inputs = net->input_layer + INPUT_OFFSET(net);
}


void
rnn_bptt_calc_deltas(RecurNN *net, int accumulate_delta)
{
  RecurNNBPTT *bptt = net->bptt;
  RecurExtraLayer *bottom = net->bottom_layer;
  float *bottom_error = (bottom) ? bottom->o_error : NULL;

  /*top layer */
  if (! accumulate_delta){
    zero_aligned_array(bptt->ho_delta, net->ho_size);
  }
  float top_error_sum = backprop_top_layer(net);
  float top_error_scaled = softclip_scale(top_error_sum,
      net->h_size * MAX_TOP_ERROR_FACTOR, bptt->h_error, net->h_size);

  single_layer_sgd(net->hidden_layer, net->h_size, bptt->o_error, net->o_size,
      bptt->ho_delta);

  /*recurrent layer. Both accumulating and non-accumulating branches are
    complicated by emergency scaling requirements.*/
  float bptt_error_sum;
  if (accumulate_delta){
    zero_aligned_array(bptt->ih_delta_tmp, net->ih_size);
    bptt_error_sum = bptt_and_accumulate_error(net,
        bptt->ih_delta_tmp, bottom_error, top_error_scaled);
    add_aligned_arrays(bptt->ih_delta, net->ih_size, bptt->ih_delta_tmp,
        bptt->ih_scale);
  }
  else {
    zero_aligned_array(bptt->ih_delta, net->ih_size);
    bptt_error_sum = bptt_and_accumulate_error(net,
        bptt->ih_delta, bottom_error, top_error_scaled);
    if (bptt->ih_scale != 1.0f){
      scale_aligned_array(bptt->ih_delta, net->ih_size, bptt->ih_scale);
    }
  }

  /*bottom layer */
  if (bottom){
    if (! accumulate_delta){
      zero_aligned_array(bottom->delta, bottom->i_size * bottom->o_size);
    }
    single_layer_sgd(bottom->inputs, bottom->i_size, bottom_error, bottom->o_size,
        bottom->delta);
    if (net->log){
      float be = 0;
      for (int i = 0; i < bottom->output_size; i++){
        be += fabsf(bottom_error[i]);
      }
      rnn_log_float(net, "bottom_error", be);
    }
  }
  net->generation++;
  if (net->log){
    rnn_log_float(net, "error_gain", bptt_error_sum / (top_error_scaled + 1e-6));
    rnn_log_float(net, "top_error_scaled", top_error_scaled);
    rnn_log_float(net, "top_error_raw", top_error_sum);
    rnn_log_int(net, "generation", net->generation);
  }
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
        if (col >= 1 && col < INPUT_OFFSET(net)){
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

static inline void
weight_noise(rand_ctx *rng, float *weights, int width, int stride, int height,
    float deviation){
  for (int y = 0; y < height; y++){
    float *row = weights + y * stride;
    for (int x = 0; x < width; x++){
      float noise = cheap_gaussian_noise(rng) * deviation;
      row[x] += noise;
    }
  }
}

void
rnn_weight_noise(RecurNN *net, float deviation){

  weight_noise(&net->rng, net->ih_weights + 1, net->hidden_size,
      net->h_size, net->hidden_size + 1 + net->input_size,
      deviation);

  weight_noise(&net->rng, net->ho_weights, net->output_size,
      net->o_size, net->hidden_size + 1,
      deviation);

  if (net->bottom_layer){
    RecurExtraLayer *bl = net->bottom_layer;
    weight_noise(&net->rng, bl->weights + 1, bl->input_size,
        bl->i_size, bl->output_size,
        deviation);
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
}


/************************************************************************/
/*Simplified and optimised pathways for the case of a single net, possibly
  using diachronic batching.
 */

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

  hiddens[0] = 1.0f;
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

static inline float
apply_sgd_with_bptt(RecurNN *net, float top_error_sum){
  RecurNNBPTT *bptt = net->bptt;
  zero_aligned_array(bptt->ih_delta, net->ih_size);
  float error_sum = bptt_and_accumulate_error(net, bptt->ih_delta, NULL, top_error_sum);
  float rate = bptt->learn_rate * bptt->ih_scale;

  apply_learning_with_momentum(net->ih_weights, bptt->ih_delta, bptt->ih_momentum,
      net->ih_size, rate, bptt->momentum, bptt->momentum_weight);
  return error_sum;
}

static inline float
apply_sgd_with_bptt_batch(RecurNN *net, float top_error_sum, uint batch_size){
  RecurNNBPTT *bptt = net->bptt;
  float rate = bptt->learn_rate;

  zero_aligned_array(bptt->ih_delta_tmp, net->ih_size);
  float error_sum = bptt_and_accumulate_error(net, bptt->ih_delta_tmp, NULL,
      top_error_sum);

  add_aligned_arrays(bptt->ih_delta, net->ih_size, bptt->ih_delta_tmp, bptt->ih_scale);

  if ((net->generation % batch_size) == 0){
    apply_learning_with_momentum(net->ih_weights, bptt->ih_delta, bptt->ih_momentum,
        net->ih_size, rate, bptt->momentum, bptt->momentum_weight);
    zero_aligned_array(bptt->ih_delta, net->ih_size);
  }
  return error_sum;
}

void
rnn_bptt_calculate(RecurNN *net, uint batch_size){
  float bptt_error_sum;
  float top_error_sum = apply_sgd_top_layer(net);
  float top_error_scaled = softclip_scale(top_error_sum,
      net->h_size * MAX_TOP_ERROR_FACTOR, net->bptt->h_error, net->h_size);

  if (batch_size > 1)
    bptt_error_sum = apply_sgd_with_bptt_batch(net, top_error_scaled, batch_size);
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




/*rnn_scale_initial_weights() tries to scale the weights to give approximately
  the right gain */

void
rnn_scale_initial_weights(RecurNN *net, float target_gain){
  int h_size = net->h_size;
  float layer_in[h_size];
  float layer_out[h_size];
  int i;
  double net_adjustment = 1.0;
  double tail_in = 0;
  double tail_out = 0;
  const double generations = 10000;
  for (double j = 1; j < generations; j++){
    float sum_out;
    float sum_in = 1;
    layer_in[0] = 1;
    for (i = 1; i < net->hidden_size; i++){
      float n = MAX(cheap_gaussian_noise(&net->rng), 0);
      layer_in[i] = n;
      sum_in += n * n;
    }
    for (i = net->hidden_size; i < h_size; i++){
      layer_in[i] = 0;
      layer_out[i] = 0;
    }
    calculate_interlayer(layer_in,
        net->hidden_size + 1,
        layer_out,
        h_size,
        net->ih_weights);
    layer_out[0] = 1.0f;
    sum_out = 0;
    for (i = 0; i < net->hidden_size; i++){
      float h = layer_out[i];
      h = (h > 0.0f) ? h : 0.0f;
      layer_out[i] = h;
      sum_out += h * h;
    }
    double ratio = sum_out / sum_in;
    double adj = (target_gain * 10 + j) / (ratio * 10 + j);
    net_adjustment *= adj;
    scale_aligned_array(net->ih_weights, net->ih_size, adj);
    MAYBE_DEBUG("j %f sum in %.2f out %.2f, ratio %.2f adj %.2f net adj %.2f",
        j, sum_in, sum_out, ratio, adj, net_adjustment);
    if (j > generations * 0.95){
      tail_in += sum_in;
      tail_out += sum_out;
    }
  }
  DEBUG("scaled toward target gain %.3f; hit roughly %.3f; adjusted by %.3f",
      target_gain, tail_out / tail_in, net_adjustment);
}
