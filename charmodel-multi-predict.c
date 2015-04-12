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
  int i;
  float *restrict answer = one_hot_opinion(net, c, net->presynaptic_noise);
  int n_classes = net->output_size / alphabet_len;
  float err = 0;
  u64 threshold = leakage * UINT64_MAX;
  memset(error, 0, net->output_size * sizeof(float));
  for (i = 0; i < n_classes; i ++){
    if (i == target_class){
      softmax_best_guess(error, answer, alphabet_len);
      error[next] += 1.0f;
      err = error[next];
    }
    else if (rand64(&net->rng) < threshold){
      softmax_best_guess(error, answer, alphabet_len);
      error[next] += 1.0f;
    }
    error += alphabet_len;
    answer += alphabet_len;
  }
  return err;
}

#define INNER_CYCLE_PGM_DUMP()                                  \
  if (input_ppm){                                               \
    temporal_ppm_add_row(input_ppm, net->input_layer);          \
  }                                                             \
  if (error_ppm){                                               \
    temporal_ppm_add_row(error_ppm, net->bptt->o_error);        \
  }                                                             \
  if (periodic_pgm_period) {                                    \
    if (! --periodic_pgm_countdown){                            \
      periodic_pgm_countdown = periodic_pgm_period;             \
      rnn_multi_pgm_dump(net, periodic_pgm_string,              \
          "multi-text");                                        \
    }                                                           \
  }


#define INNER_CYCLE_REPORTING()                                 \
  if (report){                                                  \
    error += e;                                                 \
    entropy += capped_log2f(1.0f - e);                          \
  }                                                             \
  INNER_CYCLE_PGM_DUMP()


static inline void
text_train_plain_sgd(RecurNN *net, u8 *text, int len,
    int target_class, int batch_size, float leakage,
    int alphabet_len, RnnCharProgressReport *report,
    TemporalPPM *input_ppm, TemporalPPM *error_ppm,
    const char *periodic_pgm_string, int periodic_pgm_period,
    int periodic_pgm_countdown)
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

static inline void
text_train_fancy(RecurNN *net, u8 *text, int len, int learning_style,
    int target_class, int batch_size, float leakage,
    int alphabet_len, RnnCharProgressReport *report,
    TemporalPPM *input_ppm, TemporalPPM *error_ppm,
    const char *periodic_pgm_string, int periodic_pgm_period,
    int periodic_pgm_countdown)
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

void
rnn_char_multitext_spin(RecurNN *net, u8 *text, int len,
    TemporalPPM *input_ppm, TemporalPPM *error_ppm,
    const char *periodic_pgm_string, int periodic_pgm_period)
{
  int periodic_pgm_countdown;
  if (periodic_pgm_period){
    periodic_pgm_countdown = (periodic_pgm_period -
        net->generation % periodic_pgm_period);
  }
  else {
    periodic_pgm_countdown = 0;
  }
  if (error_ppm){
    memset(net->bptt->o_error, 0, net->output_size * sizeof(float));
  }
  for(int i = 0; i < len; i++){
    rnn_bptt_advance(net);
    one_hot_opinion(net, text[i], net->presynaptic_noise);
    INNER_CYCLE_PGM_DUMP();
  }
}

#undef INNER_CYCLE_REPORTING
#undef INNER_CYCLE_PGM_DUMP

/* stochastic leakage? */

void
rnn_char_multitext_train(RecurNN *net, u8 *text, int len, int alphabet_len,
    int target_class, float leakage, RnnCharProgressReport *report,
    int learning_style, float momentum, int batch_size,
    TemporalPPM *input_ppm, TemporalPPM *error_ppm,
    const char *periodic_pgm_string, int periodic_pgm_period)
{
  struct timespec time_start;
  struct timespec time_end;
  int periodic_pgm_countdown;
  if (periodic_pgm_period){
    periodic_pgm_countdown = (periodic_pgm_period -
        net->generation % periodic_pgm_period);
  }
  else {
    periodic_pgm_countdown = 0;
  }

  batch_size = MAX(batch_size, 1); /*XXX adjust for len */

  if (report){
    clock_gettime(CLOCK_MONOTONIC, &time_start);
  }
  if (learning_style == RNN_MOMENTUM_WEIGHTED){
    text_train_plain_sgd(net, text, len, target_class, batch_size,
        leakage, alphabet_len, report,
        input_ppm, error_ppm,
        periodic_pgm_string, periodic_pgm_period,
        periodic_pgm_countdown);
  }
  else {
    text_train_fancy(net, text, len, learning_style,
        target_class, batch_size, leakage, alphabet_len,
        report, input_ppm, error_ppm,
        periodic_pgm_string, periodic_pgm_period,
        periodic_pgm_countdown);
  }
  if (report){
    clock_gettime(CLOCK_MONOTONIC, &time_end);
    s64 secs = time_end.tv_sec - time_start.tv_sec;
    s64 nano = time_end.tv_nsec - time_start.tv_nsec;
    double elapsed = secs + 1e-9 * nano;
    report->per_second = (len - 1) / elapsed;
  }
}


void
rnn_char_multi_cross_entropy(RecurNN *net, const u8 *text, int len,
    int alphabet_len, double *entropy, int ignore_start){
  float error[alphabet_len];
  int i, j;
  int n_classes = net->output_size / alphabet_len;
  /*skip the first few because state depends too much on previous experience */
  for (i = 0; i < ignore_start; i++){
    one_hot_opinion(net, text[i], 0);
  }
  for (; i < len - 1; i++){
    float *restrict answer = one_hot_opinion(net, text[i], 0);
    for (j = 0; j < n_classes; j++){
      float *group = answer + alphabet_len * j;
      softmax(error, group, alphabet_len);
      float e = error[text[i + 1]];
      entropy[j] -= capped_log2f(e);
    }
  }
  for (j = 0; j < n_classes; j++){
    entropy[j] /= (len - ignore_start - 1);
  }
}
