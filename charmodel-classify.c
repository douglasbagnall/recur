/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL

Classify text, presumably by language.
*/

#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include <math.h>
#include "path.h"
#include "badmaths.h"
#include <stdio.h>

#include "charmodel.h"
#include "charmodel-helpers.h"
#include "utf8.h"

static inline int
next_all_ones(int x){
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x;
}

#define NO_CLASS 0xFF

/*Adjust the lag of the text predictions in-place. Anything that gets adjusted
  out of range gets its target set to 0xff, which indicates no training */

void
rnn_char_adjust_text_lag(RnnCharClassifiedChar *text, int len, int lag){
  int i;
  /*lag could be positive or negative*/
  DEBUG("text %d %d, %d %d... len %d, lag %d",
      text[0].class, text[0].symbol, text[1].class, text[1].symbol,
      len, lag);
  if (lag > 0){
    for (i = len - 1; i >= lag; i--){
      text[i].class = text[i - lag].class;
    }
    for (; i >= 0; i--){
      text[i].class = NO_CLASS;
    }
  }
  else if (lag < 0){
    for (i = 0; i < lag + len; i++){
      text[i].class = text[i - lag].class;
    }
    for (; i < len; i++){
      text[i].class = NO_CLASS;
    }
  }
}


static double
get_elapsed_interval(struct timespec **start, struct timespec **end){
  clock_gettime(CLOCK_MONOTONIC, *end);
  s64 secs = (*end)->tv_sec - (*start)->tv_sec;
  s64 nano = (*end)->tv_nsec - (*start)->tv_nsec;
  double elapsed = secs + 1e-9 * nano;
  struct timespec *tmp = *end;
  *end = *start;
  *start = tmp;
  return elapsed;
}

/* lag needs to be preadjusted */
int
rnn_char_classify_epoch(RnnCharClassifier *model, const RnnCharClassifiedChar *text,
    int len, int start, int stop, int ignore_start){
  int n_nets = model->n_training_nets;
  int spacing = (len - 1) / n_nets;
  int correct = 0;
  float mean_error = 0;
  float entropy = 0;
  RecurNN *net = model->net;
  RecurNN **nets = model->training_nets;
  uint report_counter = net->generation % model->report_interval;
  int end = (stop < len) ? stop : len;
  struct timespec timers[2];
  struct timespec *time_start = timers;
  struct timespec *time_end = timers + 1;
  for (int i = MAX(0, start); i < end; i++){
    float momentum = rnn_calculate_momentum_soft_start(net->generation,
        model->momentum, model->momentum_soft_start);

    int offset = i;
    for (int j = 0; j < n_nets; j++){
      RnnCharClassifiedChar t = text[offset];
      RecurNN *n = nets[j];
      rnn_bptt_advance(n);
      float *answer = one_hot_opinion(net, t.symbol);
      if (t.class != NO_CLASS){
        float *error = n->bptt->o_error;
        ASSUME_ALIGNED(error);
        int winner = softmax_best_guess(error, answer, net->output_size);
        correct += (winner == t.class);
        float e = error[t.class] + 1.0f;
        error[t.class] = e;
        mean_error += e;
        entropy += capped_log2f(1.0f - e);
      }
      rnn_bptt_calc_deltas(n, j ? 1 : 0);

      offset += spacing;
      if (offset >= len - 1){
        offset -= len - 1;
      }
    }
    if (ignore_start){
      ignore_start--;
    }
    else {
      rnn_apply_learning(net, model->momentum_style, momentum);
    }
    report_counter++;
    if (report_counter == model->report_interval){
      report_counter = 0;
      double elapsed = get_elapsed_interval(&time_start, &time_end);
      float scale = 1.0f / (model->report_interval * n_nets);
      entropy *= -scale;
      mean_error *= scale;
      float accuracy = correct * scale;
      double per_sec = 1.0 / scale / elapsed;
      rnn_log_float(net, "t_error", mean_error);
      rnn_log_float(net, "t_entropy", entropy);
      rnn_log_float(net, "momentum", net->bptt->momentum);
      rnn_log_float(net, "accuracy", accuracy);
      rnn_log_float(net, "learn-rate", net->bptt->learn_rate);
      rnn_log_float(net, "per_second", per_sec);
      correct = 0;
      mean_error = 0.0f;
      entropy = 0.0f;
    }
  }
  if (stop && (int)net->generation >= stop){
    return 1;
  }
  return 0;
}
