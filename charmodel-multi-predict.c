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


static void
accumulate_ranges(RecurErrorRange *dest, const RecurErrorRange *src, int max_n)
{
  int end;
  int i;
  RecurErrorRange src2[max_n];
  const RecurErrorRange *tmp, *a, *b;
  if (src->start < 0) { /* merging empty list */
    return;
  }

  if (dest->start < 0) {  /* merging into empty list */
    for (i = 0; i < max_n; i++) {
      dest[i] = src[i];
      if (dest[i].start < 0) {
        return;
      }
    }
  }

  for (i = 0; i < max_n; i++) {
    src2[i] = dest[i];
    if (src2[i].start < 0) {
      break;
    }
  }

  end = -1;
  a = src;
  b = src2;
  dest --;
  /* sort of like mergesort. a is always the leader */
  for (i = 0; i < max_n; i++){
    if (a->start < 0) {
      if (b->start < 0) {
        break;
      }
      /* a is finished but b isn't. swap them. */
      tmp = b;
      b = a;
      a = tmp;
    }
    else if (b->start >= 0 && b->start < a->start) {
      tmp = b;
      b = a;
      a = tmp;
    }

    if (a->start > end) {
      /* we start a new dest node */
      dest++;
      dest->start = a->start;
      dest->len = a->len;
      end = a->start + a->len;
    }
    else {
      /* we merge it into dest */
      end = a->start + a->len;
      dest->len = end - dest->start;
    }
    a++;
  }
  dest++;
  dest->len = 0;
  dest->start = -1;
}



static inline float
multi_softmax_error(RecurNN *net, float *restrict error, int c, int next,
    int target_class, int alphabet_len, float leakage,
    RecurErrorRange *error_ranges)
{
  int i;
  float *restrict answer = one_hot_opinion_sparse(net, c,
      net->presynaptic_noise, error_ranges);
  int n_classes = net->output_size / alphabet_len;
  float err = 0;
  int j = 0;
  int k;
  u64 threshold = leakage * UINT64_MAX;

  for (i = 0; i < n_classes; i++){
    int offset = i * alphabet_len;
    if (i == target_class || rand64(&net->rng) < threshold){
      int range_start = ALIGNED_ROUND_DOWN(offset);
      int range_end = ALIGNED_ROUND_UP(offset + alphabet_len);

      for (k = range_start; k < offset; k++){
        error[k] = 0.0f;
      }

      softmax_best_guess(error + offset, answer + offset, alphabet_len);
      error[offset + next] += 1.0f;

      for (k = offset + alphabet_len; k < range_end; k++){
        error[k] = 0.0f;
      }

      if (i == target_class){
        err = error[offset + next];
      }

      if (j){ /* merge adjacent ranges */
        RecurErrorRange prev = error_ranges[j - 1];
        if (prev.start + prev.len >= range_start){
          error_ranges[j - 1].len = range_end - prev.start;
          continue;
        }
      }

      error_ranges[j].start = range_start;
      error_ranges[j].len = range_end - range_start;
      j++;
    }
  }
  error_ranges[j].start = -1;
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
  int n_classes = net->output_size / alphabet_len;
  RecurErrorRange top_error_ranges[n_classes + 1];
  RecurErrorRange top_delta_ranges[n_classes + 1];
  int countdown = batch_size - net->generation % batch_size;

  for(i = 0; i < len - 1; i++, countdown--){
    rnn_bptt_advance(net);
    float e = multi_softmax_error(net, bptt->o_error, text[i], text[i + 1],
        target_class, alphabet_len, leakage, top_error_ranges);
    if (countdown == 0){
      /* top_error_range learning only implemented for adagrad (so far) */
      if (learning_style == RNN_ADAGRAD){
        rnn_apply_learning(net, learning_style, bptt->momentum,
            NULL);
      }
      else {
        rnn_apply_learning(net, learning_style, bptt->momentum, NULL);
      }
      countdown = batch_size;
      rnn_bptt_calc_deltas(net, 0, top_error_ranges);
    }
    else {
      rnn_bptt_calc_deltas(net, 1, top_error_ranges);
      accumulate_ranges(top_delta_ranges, top_error_ranges, n_classes + 1);
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
    int hot = text[i];
    one_hot_opinion_sparse(net, hot, net->presynaptic_noise, NULL);
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
    int hot = text[i];
    one_hot_opinion_sparse(net, hot, 0, NULL);
  }
  for (; i < len - 1; i++){
    int hot = text[i];
    float *restrict answer = one_hot_opinion_sparse(net, hot, 0, NULL);
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
