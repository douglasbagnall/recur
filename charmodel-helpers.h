#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include <math.h>
#include "path.h"
#include "badmaths.h"
#include "charmodel.h"

static inline float
capped_log2f(float x){
  return (x < 1e-30f) ? -100.0f : log2f(x);
}

static inline float*
one_hot_opinion(RecurNN *net, int hot, float presynaptic_noise){
  float *inputs;
  int len;
  if (net->bottom_layer){
    inputs = net->bottom_layer->inputs;
    len = net->bottom_layer->input_size;
  }
  else{
    inputs = net->real_inputs;
    len = net->input_size;
  }

  memset(inputs, 0, len * sizeof(float));
  inputs[hot] = 1.0f;
  return rnn_opinion(net, NULL, presynaptic_noise, NULL);
}

static inline float*
one_hot_opinion_sparse(RecurNN *net, int hot,
    float presynaptic_noise, RecurErrorRange *ranges){
  float *inputs;
  int len;
  if (net->bottom_layer){
    inputs = net->bottom_layer->inputs;
    len = net->bottom_layer->input_size;
  }
  else{
    inputs = net->real_inputs;
    len = net->input_size;
  }

  memset(inputs, 0, len * sizeof(float));
  inputs[hot] = 1.0f;
  return rnn_opinion(net, NULL, presynaptic_noise, ranges);
}

static inline float*
one_hot_opinion_with_cold(RecurNN *net, int hot, int cold,
    float presynaptic_noise){
  /* This version assumes that the input array is already
     zero except for the point named in <cold> */
  float *inputs;
  if (net->bottom_layer){
    inputs = net->bottom_layer->inputs;
  }
  else{
    inputs = net->real_inputs;
  }
  /*XXX not checking ranges!*/
  inputs[cold] = 0.0f;
  inputs[hot] = 1.0f;
  return rnn_opinion(net, NULL, presynaptic_noise, NULL);
}


/* This helps one_hot_opinion_with_cold() by zeroing the array at the
   beginning of each cycle */

static inline void
zero_real_inputs(RecurNN *net){
  float *inputs;
  int len;
  if (net->bottom_layer){
    inputs = net->bottom_layer->inputs;
    len = net->bottom_layer->input_size;
  }
  else{
    inputs = net->real_inputs;
    len = net->input_size;
  }
  memset(inputs, 0, len * sizeof(float));
}

