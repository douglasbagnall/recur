/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL

This uses the RNN to predict the next character in a text sequence.
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
#include "colour.h"


static inline float
multi_softmax_error(RecurNN *net, float *restrict error, int c, int next,
    int target_class, int alphabet_len, float leakage)
{
  int i, j;
  float *restrict answer = one_hot_opinion(net, c, net->presynaptic_noise);
  int n_classes = net->output_size / alphabet_len;
  float err = 0;

  for (i = 0; i < n_classes; i ++){
    /*XXX also try stochastic leakage, randomly zeroing instead of reducing */
    softmax_best_guess(error, answer, alphabet_len);
    error[next] += 1.0f;
    if (i != target_class){
      for (j = 0; j < alphabet_len; j++){
        error[j] *= leakage;
      }
    }
    else {
      err = error[next];
    }
    error += alphabet_len;
    answer += alphabet_len;
  }
  return err;
}



#define INNER_CYCLE_REPORTING()                                 \
  if (report){                                                  \
    error += e;                                                 \
    entropy += capped_log2f(1.0f - e);                          \
  }                                                             \
  if (input_ppm){                                               \
    temporal_ppm_add_row(input_ppm, net->input_layer);          \
  }                                                             \
  if (error_ppm){                                               \
    temporal_ppm_add_row(error_ppm, net->bptt->o_error);        \
  }


static void
text_train_plain_sgd(RecurNN *net, u8 *text, int len,
    int target_class, int batch_size, float leakage,
    int alphabet_len, RnnCharProgressReport *report,
    TemporalPPM *input_ppm, TemporalPPM *error_ppm)
{
  int i;
  float error = 0.0f;
  float entropy = 0.0f;
  RecurNNBPTT *bptt = net->bptt;
  for(i = 0; i < len - 1; i++){
    rnn_bptt_advance(net);
    float e = multi_softmax_error(net, bptt->o_error, text[i], text[i + 1],
        target_class, alphabet_len, leakage);
    rnn_bptt_calculate(net, batch_size);
    INNER_CYCLE_REPORTING();
  }
  if (report){
    float report_scale = 1.0f / (len - 1);
    report->training_entropy = -entropy * report_scale;
    report->training_error = error * report_scale;
  }
}


static void
text_train_fancy(RecurNN *net, u8 *text, int len, int learning_style,
    int target_class, int batch_size, float leakage,
    int alphabet_len, RnnCharProgressReport *report,
    TemporalPPM *input_ppm, TemporalPPM *error_ppm)
{
  int i;
  float error = 0.0f;
  float entropy = 0.0f;
  RecurNNBPTT *bptt = net->bptt;
  int countdown = batch_size - net->generation % batch_size;
  for(i = 0; i < len - 1; i++, countdown--){
    rnn_bptt_advance(net);
    float e = multi_softmax_error(net, bptt->o_error, text[i], text[i + 1],
        target_class, alphabet_len, leakage);
    if (countdown == 0){
      rnn_apply_learning(net, learning_style, bptt->momentum);
      countdown = batch_size;
      rnn_bptt_calc_deltas(net, 0);
    }
    else {
      rnn_bptt_calc_deltas(net, 1);
    }
    INNER_CYCLE_REPORTING();
  }
  if (report){
    float report_scale = 1.0f / (len - 1);
    report->training_entropy = -entropy * report_scale;
    report->training_error = error * report_scale;
  }
}

#undef INNER_CYCLE_REPORTING

void
rnn_char_text_train(RnnCharModel *model, u8 *text, int len,
    int target_class, float leakage, RnnCharProgressReport *report)
{
  RecurNN *net = model->net;
  RecurNNBPTT *bptt = net->bptt;
  struct timespec time_start;
  struct timespec time_end;
  int batch_size = batch_size; /*XXX adjust for len */
  if (report){
    clock_gettime(CLOCK_MONOTONIC, &time_start);
  }
  if (model->learning_style == RNN_MOMENTUM_WEIGHTED){
    bptt->momentum = rnn_calculate_momentum_soft_start(net->generation,
        model->momentum, model->momentum_soft_start);
    text_train_plain_sgd(net, text, len, target_class, batch_size,
        leakage, model->alphabet->len, report,
        model->images.input_ppm, model->images.error_ppm);
  }
  else {
    text_train_fancy(net, text, len, model->learning_style,
        target_class, batch_size, leakage, model->alphabet->len,
        report, model->images.input_ppm, model->images.error_ppm);
  }
  if (report){
    clock_gettime(CLOCK_MONOTONIC, &time_end);
    s64 secs = time_end.tv_sec - time_start.tv_sec;
    s64 nano = time_end.tv_nsec - time_start.tv_nsec;
    double elapsed = secs + 1e-9 * nano;
    report->per_second = (len - 1) / elapsed;
  }
}
