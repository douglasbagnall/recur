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
    int target_class, int alphabet_len, float leakage,
    RecurErrorRange *error_ranges)
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

  for (i = 0; i < n_classes; i++){
    int offset = i * alphabet_len;
    if (i == target_class || rand64(&net->rng) < threshold){
      softmax_best_guess(error + offset, answer + offset, alphabet_len);
      error[offset + next] += 1.0f;
      if (i == target_class){
        err = error[offset + next];
      }
      int range_start = ALIGNED_ROUND_DOWN(offset);
      int range_end = ALIGNED_ROUND_UP(offset + alphabet_len);
      if (j){
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


RnnCharMultiConfab *
rnn_char_new_multi_confab(RecurNN *net, RnnCharAlphabet *alphabet, int n_classes,
    int target_len, uint confab_period)
{
  int len = target_len / n_classes - 1;
  if (len < 1) {
    DEBUG("no room to confabulate %d sub-models in %d characters",
        n_classes, target_len);
    return NULL;
  }

  RnnCharMultiConfab *mc = calloc(1, sizeof(*mc));
  mc->char_len = len;
  mc->byte_len = len * 6 + 1;
  mc->alphabet = alphabet;
  mc->period = confab_period;
  mc->n_classes = n_classes;

  mc->last_char = calloc(n_classes, sizeof(int));
  mc->strings = calloc(n_classes, sizeof(char *));
  mc->nets = calloc(n_classes, sizeof(RecurNN *));
  for (int i = 0; i < n_classes; i++) {
    mc->strings[i] = calloc(mc->byte_len, 1);
    mc->nets[i] = rnn_clone(net,
        net->flags & ~(RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS),
        RECUR_RNG_SUBSEED,
        NULL);
  }
  return mc;
}

void
rnn_char_free_multi_confab(RnnCharMultiConfab *mc)
{
  for (int i = 0; i < mc->n_classes; i++) {
    free(mc->strings[i]);
    rnn_delete_net(mc->nets[i]);
  }
  free(mc->strings);
  free(mc->nets);
  free(mc->last_char);
  free(mc);
}


static inline int
offset_guess_next_character(RecurNN *net, float *error, int hot, float bias,
    uint offset, int alphabet_len)
{
  uint i;
  float r;
  float *answer = one_hot_opinion(net, hot, 0);
  float *group = answer + alphabet_len * offset;

  biased_softmax(error, group, alphabet_len, bias);
  for(;;){
    r = rand_double(&net->rng);
    float accum = 0.0;
    for (i = 0; i < alphabet_len; i++){
      accum += error[i];
      if (r < accum){
        return i;
      }
    }
  }
}


static int
multi_confab(RnnCharMultiConfab *mc)
{
  const int *alphabet = mc->alphabet->points;
  int total = 0;
  bool utf8 = mc->alphabet->flags & RNN_CHAR_FLAG_UTF8;
  float error[mc->alphabet->len];

  int char_width = utf8 ? 5 : 1;
  if (mc->byte_len <= char_width) {
    DEBUG("insufficient space to confabulate (%d bytes)", mc->byte_len);
    return 0;
  }
  for (uint m = 0; m < mc->n_classes; m++) {
    RecurNN *net = mc->nets[m];
    char *d = mc->strings[m];
    int n = mc->last_char[m];
    int bytes_left = mc->byte_len;
    for (int i = 0;
         i < mc->char_len && bytes_left > char_width;
         i++){
        n = offset_guess_next_character(net, error, n, mc->bias,
            m, mc->alphabet->len);
      int w = write_possibly_utf8_char(alphabet[n], d, utf8);
      d += w;
      bytes_left -= w;
    }
    *d = '\0';
    total += mc->byte_len - bytes_left;
    mc->last_char[m] = n;
  }
  return total;
}


static int
multi_confab_format_line(RnnCharMultiConfab *mc, char *dest, int len, char *sep)
{
  int i = 0;
  bool in_sep = false;
  int total;

  char *src = mc->strings[0];

  for (total = 0; total < len - 1; total++, src++) {
    char c = *src;
    while (c == '\0') {
      in_sep = ! in_sep;
      if (in_sep) {
        src = sep;
      }
      else {
        i++;
        if (i >= mc->n_classes) {
          break;
        }
        src = mc->strings[i];
      }
      c = *src;
    }
    dest[total] = c;
  }
  dest[total] = '\0';
  return total;
}




static inline void
text_train(RecurNN *net, u8 *text, int len, int learning_style,
    int target_class, int batch_size, float leakage,
    int alphabet_len, RnnCharProgressReport *report,
    RnnCharMultiConfab *mc,
    TemporalPPM *input_ppm, TemporalPPM *error_ppm,
    const char *periodic_pgm_string, int periodic_pgm_period,
    int periodic_pgm_countdown)
{
  int i;
  float error = 0.0f;
  float entropy = 0.0f;
  RecurNNBPTT *bptt = net->bptt;
  int n_classes = net->output_size / alphabet_len;
  RecurErrorRange top_error_ranges[n_classes];
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
  if (report) {
    float report_scale = 1.0f / (len - 1);
    report->training_entropy = -entropy * report_scale;
    report->training_error = error * report_scale;
  }
  if (mc && mc->period && i % mc->period == 0) {
    char confab_line[mc->byte_len];
    multi_confab(mc);
    multi_confab_format_line(mc, confab_line, mc->byte_len,
        C_CYAN "|" C_NORMAL);
    printf("%8u" C_CYAN "|" C_NORMAL "%s\n", net->generation, confab_line);
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
    RnnCharMultiConfab *confab,
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
      report, confab, input_ppm, error_ppm,
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
