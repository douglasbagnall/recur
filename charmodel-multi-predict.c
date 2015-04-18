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


static inline int
add_or_adjust_error_range(int *ranges, int alphabet_len, int i, int j){
  int start = i * alphabet_len;
  int end = start + alphabet_len;
  ASSUME_ALIGNED_LENGTH(start);
  ALIGNED_LENGTH_ROUND_UP(end);

  if (j && ranges[j - 1] >= start){
    ranges[j - 1] = end;
    return j;
  }
  ranges[j] = start;
  ranges[j + 1] = end;
  return j + 2;
}


static inline float
multi_softmax_error(RecurNN *net, float *restrict error, int c, int next,
    int target_class, int alphabet_len, float leakage, int *error_ranges)
{
  int i;
  float *restrict answer = one_hot_opinion(net, c, net->presynaptic_noise);
  int n_classes = net->output_size / alphabet_len;
  float err = 0;
  int j = 0;
  u64 threshold = leakage * UINT64_MAX;
  /* XXX memset is *almost* redundant with error_ranges, but the zeros are
     necessary for alignment, and are useful for debug images. */
  memset(error, 0, net->output_size * sizeof(float));
  for (i = 0; i < n_classes; i ++){
    if (i == target_class){
      softmax_best_guess(error, answer, alphabet_len);
      error[next] += 1.0f;
      err = error[next];
      j = add_or_adjust_error_range(error_ranges, alphabet_len, i, j);
    }
    else if (rand64(&net->rng) < threshold){
      softmax_best_guess(error, answer, alphabet_len);
      error[next] += 1.0f;
      j = add_or_adjust_error_range(error_ranges, alphabet_len, i, j);
    }
    error += alphabet_len;
    answer += alphabet_len;
  }
  error_ranges[j] = -1;
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


static inline void
text_train(RecurNN *net, u8 *text, int len, int learning_style,
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
  int top_error_ranges[net->output_size / alphabet_len * 2 + 2];
  int countdown = batch_size - net->generation % batch_size;
  for(i = 0; i < len - 1; i++, countdown--){
    rnn_bptt_advance(net);
    float e = multi_softmax_error(net, bptt->o_error, text[i], text[i + 1],
        target_class, alphabet_len, leakage, top_error_ranges);
    if (countdown == 0){
      rnn_apply_learning(net, learning_style, bptt->momentum);
      countdown = batch_size;
      rnn_bptt_calc_deltas(net, 0, top_error_ranges);
    }
    else {
      rnn_bptt_calc_deltas(net, 1, top_error_ranges);
    }
    if (report){
      error += e;
      entropy += capped_log2f(1.0f - e);
    }
    INNER_CYCLE_PGM_DUMP();
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
  text_train(net, text, len, learning_style,
      target_class, batch_size, leakage, alphabet_len,
      report, input_ppm, error_ppm,
      periodic_pgm_string, periodic_pgm_period,
      periodic_pgm_countdown);

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
