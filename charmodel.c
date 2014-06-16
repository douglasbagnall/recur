/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL

This uses the RNN to predict the next character in a text sequence.
*/

#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include <math.h>
#include "path.h"
#include "badmaths.h"
//#include <errno.h>
#include <stdio.h>
//#include <fenv.h>
#include <ctype.h>

#include "charmodel.h"

static u8*
new_char_lut(const char *alphabet, const u8 *collapse_chars, int learn_caps){
  int i;
  int len = strlen(alphabet);
  int collapse_target = 0;
  int space;
  char *space_p = strchr(alphabet, ' ');
  if (space_p){
    space = space_p - alphabet;
  }
  else {
    space = 0;
    DEBUG("space is not in alphabet: %s", alphabet);
  }
  u8 *ctn = malloc_aligned_or_die(257);
  memset(ctn, space, 257);
  for (i = 0; collapse_chars[i]; i++){
    ctn[collapse_chars[i]] = collapse_target;
  }
  for (i = 0; i < len; i++){
    u8 c = alphabet[i];
    ctn[c] = i;
    if (islower(c)){
      if (learn_caps){
        ctn[c - 32] = i | 0x80;
      }
      else {
        ctn[c - 32] = i;
      }
    }
  }
  return ctn;
}

u8*
alloc_and_collapse_text(char *filename, const char *alphabet, const u8 *collapse_chars,
    long *len, int learn_caps, int quietness){
  int i, j;
  u8 *char_to_net = new_char_lut(alphabet, collapse_chars, learn_caps);
  FILE *f = fopen_or_abort(filename, "r");
  int err = fseek(f, 0, SEEK_END);
  *len = ftell(f);
  err |= fseek(f, 0, SEEK_SET);
  u8 *text = malloc(*len + 1);
  u8 prev = 0;
  u8 c;
  int chr = 0;
  int space = char_to_net[' '];
  j = 0;
  for(i = 0; i < *len && chr != EOF; i++){
    chr = getc(f);
    c = char_to_net[chr];
    if (c != space || prev != space){
      prev = c;
      text[j] = c;
      j++;
    }
  }
  text[j] = 0;
  *len = j;
  if (quietness < 1){
    STDERR_DEBUG("original text was %d chars, collapsed is %d", i, j);
  }
  err |= fclose(f);
  if (err && quietness < 2){
    STDERR_DEBUG("something went wrong with the file %p (%s). error %d",
        f, filename, err);
  }
  return text;
}

void
dump_collapsed_text(const u8 *text, int len, const char *name,
    const char *alphabet)
{
  int i;
  FILE *f = fopen_or_abort(name, "w");
  for (i = 0; i < len; i++){
    u8 c = text[i];
    if (c & 0x80){
      fputc(toupper(alphabet[c & 0x7f]), f);
    }
    else{
      fputc(alphabet[c], f);
    }
  }
  fclose(f);
}

static inline int
search_for_max(float *answer, int len){
  ASSUME_ALIGNED(answer);
  int j;
  int best_offset = 0;
  float best_score = *answer;
  for (j = 1; j < len; j++){
    if (answer[j] >= best_score){
      best_score = answer[j];
      best_offset = j;
    }
  }
  return best_offset;
}

static inline float*
one_hot_opinion(RecurNN *net, int hot, int learn_caps){
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
  if (learn_caps && (hot & 0x80)){
    inputs[len - 1] = 1.0f;
  }
  inputs[hot & 0x7f] = 1.0f;
  return rnn_opinion(net, NULL);
}

float
net_error_bptt(RecurNN *net, float *restrict error, int c, int next, int *correct,
    int learn_caps){
  ASSUME_ALIGNED(error);
  float *answer = one_hot_opinion(net, c, learn_caps);
  int winner;
  if (learn_caps){
    int len = net->output_size - 2;
    float *cap_error = error + len;
    float *cap_answer = answer + len;
    winner = softmax_best_guess(error, answer, len);
    int next_cap = (next & 0x80) ? 1 : 0;
    softmax_best_guess(cap_error, cap_answer, 2);
    cap_error[next_cap] += 1.0f;
  }
  else {
    winner = softmax_best_guess(error, answer, net->output_size);
  }
  next &= 0x7f;
  *correct = (winner == next);
  error[next] += 1.0f;
  return error[next];
}


int
opinion_deterministic(RecurNN *net, int hot, int learn_caps){
  float *answer = one_hot_opinion(net, hot, learn_caps);
  if (learn_caps){
    int len = net->output_size - 2;
    float *cap_answer = answer + len;
    int c = search_for_max(answer, len);
    int cap = search_for_max(cap_answer, 2);
    return c | (cap ? 0x80 : 0);
  }
  else {
    return search_for_max(answer, net->output_size);
  }
}

int
opinion_probabilistic(RecurNN *net, int hot, float bias, int learn_caps){
  int i;
  float r;
  float *answer = one_hot_opinion(net, hot, learn_caps);
  int n_chars = net->output_size - (learn_caps ? 2 : 0);
  int cap = 0;
  float error[net->output_size];
  if (learn_caps){
    r = rand_double(&net->rng);
    biased_softmax(error, answer + n_chars, 2, bias);
    if (r > error[0])
      cap = 0x80;
  }
  biased_softmax(error, answer, n_chars, bias);
  /*outer loop in case error doesn't quite add to 1 */
  for(;;){
    r = rand_double(&net->rng);
    float accum = 0.0;
    for (i = 0; i < n_chars; i++){
      accum += error[i];
      if (r < accum)
        return i | cap;
    }
  }
}

float
validate(RecurNN *net, const u8 *text, int len, int learn_caps){
  float error[net->output_size];
  float entropy = 0.0f;
  int i;
  int n_chars = net->output_size - (learn_caps ? 2 : 0);
  /*skip the first few because state depends too much on previous experience */
  int skip = MIN(len / 10, 5);
  for (i = 0; i < skip; i++){
    one_hot_opinion(net, text[i], learn_caps);
  }
  for (; i < len - 1; i++){
    float *answer = one_hot_opinion(net, text[i], learn_caps);
    softmax(error, answer, n_chars);
    float e = error[text[i + 1] & 0x7f];
    entropy += capped_log2f(e);
  }
  entropy /= -(len - skip - 1);
  return entropy;
}


void
init_schedule(Schedule *s, int recent_len, float margin,
    float learn_rate_min, float learn_rate_mul){
  s->recent = malloc_aligned_or_die(recent_len * sizeof(float));
  s->recent_len = recent_len;
  s->learn_rate_min = learn_rate_min;
  s->learn_rate_mul = learn_rate_mul;
  s->margin = margin;
  for (int i = 0; i < s->recent_len; i++){
    s->recent[i] = 1e10;
  }
  s->timeout = s->recent_len;
  s->eval = eval_simple;
}


void
eval_simple(Schedule *s, RecurNN *net, float score, int verbose){
  int i, j;
  RecurNNBPTT *bptt = net->bptt;
  if (bptt->learn_rate <= s->learn_rate_min){
    return;
  }
  int sample_size = s->recent_len / 3;
  i = rand_small_int(&net->rng, s->recent_len);
  s->recent[i] = score;
  if (s->timeout){
    s->timeout--;
    return;
  }
  for (++i, j = 0; j < sample_size; j++, i++){
    if (i >= s->recent_len)
      i = 0;
    if (score + s->margin < s->recent[i]){
      return;
    }
  }
  s->timeout = s->recent_len;
  bptt->learn_rate = MAX(s->learn_rate_min, bptt->learn_rate * s->learn_rate_mul);
  if (verbose){
    DEBUG("generation %7d: entropy %.4g exceeds %d recent samples (margin %.2g)."
        " setting learn_rate to %.3g. momentum %.3g",
        net->generation, score, sample_size, s->margin,
        bptt->learn_rate, net->bptt->momentum);
  }
}
