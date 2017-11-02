#ifndef HAVE_CHARMODEL_HELPERS_H
#define HAVE_CHARMODEL_HELPERS_H

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

  //XXX could just set the previous one to zero (i.e. remember it)
  memset(inputs, 0, len * sizeof(float));
  inputs[hot] = 1.0f;
  return rnn_opinion(net, NULL, presynaptic_noise);
}

#endif
