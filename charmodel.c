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

void
confabulate(RecurNN *net, char *dest, int len, const char* alphabet,
    float bias, int learn_caps){
  int i;
  static int n = 0;
  for (i = 0; i < len; i++){
    if (bias > 100)
      n = opinion_deterministic(net, n, learn_caps);
    else
      n = opinion_probabilistic(net, n, bias, learn_caps);
    int cap = n & 0x80;
    n &= 0x7f;
    int c = alphabet[n];
    if (cap){
      c = toupper(c);
    }
    dest[i] = c;
  }
}


void
init_ventropy(Ventropy *v, RecurNN *net, const u8 *text, const int len, const int lap){
  v->net = net;
  v->text = text;
  v->len = len;
  v->lap = lap;
  v->lapsize = len / lap;
  v->history = calloc(lap, sizeof(float));
  v->entropy = 0;
  v->counter = 0;
}

float
rnn_char_calc_ventropy(RnnCharModel *model, Ventropy *v, int lap)
{
  if (v->len > 0){
    if (v->lap > 1 && lap){
      v->counter++;
      if (v->counter == v->lap){
        v->counter = 0;
      }
      v->history[v->counter] = validate(v->net, v->text + v->lapsize * v->counter,
          v->lapsize, model->learn_caps);
      float sum = 0.0f;
      float div = v->lap;
      for (int j = 0; j < v->lap; j++){
        div -= v->history[j] == 0;
        sum += v->history[j];
      }
      v->entropy = div ? sum / div : 0;
    }
    else {
      v->entropy = validate(v->net, v->text, v->len, model->learn_caps);
      v->history[0] = v->entropy;
    }
  }
  return v->entropy;
}


int
epoch(RnnCharModel *model, RecurNN *confab_net, Ventropy *v,
    Schedule *schedule,
    const u8 *text, const int len,
    const int start, const int stop,
    float confab_bias, int confab_size,
    int quietness){
  int i, j;
  float error = 0.0f;
  float entropy = 0.0f;
  int correct = 0;
  float e;
  int c;
  int n_nets = model->n_training_nets;
  int spacing = (len - 1) / n_nets;
  RecurNN *net = model->net;
  RecurNN **nets = model->training_nets;
  uint report_counter = net->generation % model->report_interval;
  struct timespec timers[2];
  struct timespec *time_start = timers;
  struct timespec *time_end = timers + 1;
  clock_gettime(CLOCK_MONOTONIC, time_start);
  for(i = start; i < len - 1; i++){
    float momentum = rnn_calculate_momentum_soft_start(net->generation,
        model->momentum, model->momentum_soft_start);
    if (n_nets > 1 || model->momentum_style != RNN_MOMENTUM_WEIGHTED ||
        model->use_multi_tap_path){
      for (j = 0; j < n_nets; j++){
        RecurNN *n = nets[j];
        int offset = i + j * spacing;
        if (offset >= len - 1){
          offset -= len - 1;
        }
        rnn_bptt_advance(n);
        e = net_error_bptt(n, n->bptt->o_error,
            text[offset], text[offset + 1], &c, model->learn_caps);
        correct += c;
        error += e;
        entropy += capped_log2f(1.0f - e);
        /*Second argument to r_b_c_deltas toggles delta accumulation. Turning
          it off on the first run avoids expicit zeroing outside of the loop
          (via rnn_bptt_clear_deltas) and is thus slightly faster.
         */
        rnn_bptt_calc_deltas(n, j ? 1 : 0);
      }
      rnn_apply_learning(net, model->momentum_style, momentum);
    }
    else {
      RecurNNBPTT *bptt = net->bptt;
      bptt->momentum = momentum;
      rnn_bptt_advance(net);
      e = net_error_bptt(net, bptt->o_error, text[i], text[i + 1], &c, model->learn_caps);
      rnn_bptt_calculate(net, model->batch_size);
      correct += c;
      error += e;
      entropy += capped_log2f(1.0f - e);
    }

    if (model->input_ppm){
      temporal_ppm_add_row(model->input_ppm, net->input_layer);
    }
    if (model->error_ppm){
      temporal_ppm_add_row(model->error_ppm, net->bptt->o_error);
    }
    report_counter++;
    if (report_counter >= model->report_interval){
      report_counter = 0;
      clock_gettime(CLOCK_MONOTONIC, time_end);
      s64 secs = time_end->tv_sec - time_start->tv_sec;
      s64 nano = time_end->tv_nsec - time_start->tv_nsec;
      double elapsed = secs + 1e-9 * nano;
      struct timespec *tmp = time_end;
      time_end = time_start;
      time_start = tmp;
      float ventropy = rnn_char_calc_ventropy(model, v, 1);

      /*formerly in report_on_progress*/
      {
        float scale = 1.0f / (model->report_interval * n_nets);
        int k = net->generation >> 10;
        entropy *= -scale;
        error *= scale;
        float accuracy = correct * scale;
        double per_sec = 1.0 / scale / elapsed;
        if (confab_net && confab_size && quietness < 1){
          char confab[confab_size + 1];
          confab[confab_size] = 0;
          confabulate(confab_net, confab, confab_size, model->alphabet,
              confab_bias, model->learn_caps);
          STDERR_DEBUG("%5dk e.%02d t%.2f v%.2f a.%02d %.0f/s |%s|", k,
              (int)(error * 100 + 0.5),
              entropy, ventropy,
              (int)(accuracy * 100 + 0.5), per_sec + 0.5, confab);

        }
        rnn_log_float(net, "t_error", error);
        rnn_log_float(net, "t_entropy", entropy);
        rnn_log_float(net, "v_entropy", ventropy);
        rnn_log_float(net, "momentum", net->bptt->momentum);
        rnn_log_float(net, "accuracy", accuracy);
        rnn_log_float(net, "learn-rate", net->bptt->learn_rate);
        rnn_log_float(net, "per_second", per_sec);
        correct = 0;
        error = 0.0f;
        entropy = 0.0f;
      }

      if (model->save_net && model->filename){
        rnn_save_net(net, model->filename, 1);
      }
      if (model->periodic_pgm_dump_string){
        rnn_multi_pgm_dump(net, model->periodic_pgm_dump_string,
            model->pgm_name);
      }
      schedule->eval(schedule, net, ventropy, model->quiet < 2);
      if (model->periodic_weight_noise){
        rnn_weight_noise(net, model->periodic_weight_noise);
      }
    }
    if (stop && (int)net->generation >= stop){
      return 1;
    }
  }
  return 0;
}
