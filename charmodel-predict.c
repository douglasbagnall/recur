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
net_error_bptt(RecurNN *net, float *restrict error, int c, int next, int *correct){
  ASSUME_ALIGNED(error);
  float *answer = one_hot_opinion(net, c, net->presynaptic_noise);
  int winner;
  winner = softmax_best_guess(error, answer, net->output_size);
  *correct = (winner == next);
  error[next] += 1.0f;
  return error[next];
}

static inline int
guess_next_character(RecurNN *net, int hot, float bias){
  int i;
  float r;
  float *answer = one_hot_opinion(net, hot, 0);
  ASSUME_ALIGNED(answer);
  int len = net->output_size;
  if (bias >= 100){
    /*bias is so high as to be deterministic search for most probable */
    int best_offset = 0;
    float best_score = *answer;
    for (i = 1; i < len; i++){
      if (answer[i] >= best_score){
        best_score = answer[i];
        best_offset = i;
      }
    }
    return best_offset;
  }
  float error[net->output_size];
  biased_softmax(error, answer, len, bias);
  /*outer loop in case error doesn't quite add to 1 */
  for(;;){
    r = rand_double(&net->rng);
    float accum = 0.0;
    for (i = 0; i < len; i++){
      accum += error[i];
      if (r < accum)
        return i;
    }
  }
}

static double
get_cross_entropy(RecurNN *net, const u8 *text, int len, int skip){
  float error[net->output_size];
  double entropy = 0.0;
  int i;
  int n_chars = net->output_size;
  /*skip the first few because state depends too much on previous experience */
  for (i = 0; i < skip; i++){
    one_hot_opinion(net, text[i], 0);
  }
  for (; i < len - 1; i++){
    float *answer = one_hot_opinion(net, text[i], 0);
    softmax(error, answer, n_chars);
    float e = error[text[i + 1]];
    entropy += capped_log2f(e);
  }
  entropy /= -(len - skip - 1);
  return entropy;
}

static void
eval_simple(RnnCharModel *model, float score, int verbose){
  int i, j;
  RecurNN *net = model->net;
  RnnCharSchedule *s = &model->schedule;
  RecurNNBPTT *bptt = net->bptt;
  if (s->recent_len == 0 || bptt->learn_rate <= s->learn_rate_min){
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
    MAYBE_DEBUG("score %f vs %f", score, s->recent[i]);
    if (score < s->recent[i]){
      return;
    }
  }
  s->timeout = s->recent_len;
  bptt->learn_rate = MAX(s->learn_rate_min, bptt->learn_rate * s->learn_rate_mul);
  if (s->adjust_noise){
    net->presynaptic_noise *= s->learn_rate_mul;
    model->periodic_weight_noise *= s->learn_rate_mul;
  }
  if (verbose){
    DEBUG("generation %7d: entropy %.4g exceeds %d recent samples."
        " setting learn_rate to %.3g. momentum %.3g",
        net->generation, score, sample_size,
        bptt->learn_rate, net->bptt->momentum);
  }
}

void
rnn_char_init_schedule(RnnCharSchedule *s, int recent_len,
    float learn_rate_min, float learn_rate_mul, int adjust_noise){
  s->recent_len = recent_len;
  if (recent_len){
    s->recent = malloc_aligned_or_die(recent_len * sizeof(float));
    s->learn_rate_min = learn_rate_min;
    s->learn_rate_mul = learn_rate_mul;
    for (int i = 0; i < s->recent_len; i++){
      s->recent[i] = 1e10;
    }
  }
  s->timeout = s->recent_len;
  s->eval = eval_simple;
  s->adjust_noise = adjust_noise;
}

int
rnn_char_confabulate(RecurNN *net, char *dest, int char_len,
    int byte_len, RnnCharAlphabet* a, float bias, int stop_point){
  int i, j;
  static int n = 0;
  bool utf8 = a->flags & RNN_CHAR_FLAG_UTF8;
  const int *alphabet = a->points;
  int safe_end = byte_len - (utf8 ? 5 : 1);
  for (i = 0, j = 0; i < char_len && j < safe_end; i++){
    n = guess_next_character(net, n, bias);
    if (utf8){
      j += write_utf8_char(alphabet[n], dest + j);
    }
    else {
      dest[j] = alphabet[n];
      j++;
    }
    if (n == stop_point){
      break;
    }
  }
  dest[j] = 0;
  return j;
}


int
rnn_char_fconfab_variable(FILE *f, RecurNN *net,
    const int end_code, int max_len, RnnCharAlphabet *a, float bias){
  int i;
  static int n = 0;
  bool utf8 = a->flags & RNN_CHAR_FLAG_UTF8;
  const int *alphabet = a->points;
  for(i = 0; i < max_len; i++){
    n = guess_next_character(net, n, bias);
    if (n == end_code){
      break;
    }
    if (utf8){
      fput_utf8_char(alphabet[n], f);
    }
    else {
      fputc(alphabet[n], f);
    }
  }
  if (i == max_len){
    fputs(C_YELLOW "\\" C_NORMAL, f);
  }
  fputc('\n', f);
  //DEBUG("n %d, max_len %d, i %d", n, max_len, i);
  return i;
}

void
rnn_char_init_ventropy(RnnCharVentropy *v, RecurNN *net, const u8 *text, const int len,
    const int lap){
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
rnn_char_calc_ventropy(RnnCharModel *model, RnnCharVentropy *v, int lap)
{
  if (v->len > 0){
    if (v->lap > 1 && lap){
      v->counter++;
      if (v->counter == v->lap){
        v->counter = 0;
      }
      int skip = MIN(v->lapsize / 10, 5);
      v->history[v->counter] = get_cross_entropy(v->net,
          v->text + v->lapsize * v->counter,
          v->lapsize, skip);
      float sum = 0.0f;
      float div = v->lap;
      for (int j = 0; j < v->lap; j++){
        div -= v->history[j] == 0;
        sum += v->history[j];
      }
      v->entropy = div ? sum / div : 0;
    }
    else {
      int skip = MIN(v->len / 10, 5);
      v->entropy = get_cross_entropy(v->net, v->text, v->len, skip);
      v->history[0] = v->entropy;
    }
  }
  return v->entropy;
}


int
rnn_char_epoch(RnnCharModel *model, RecurNN *confab_net, RnnCharVentropy *v,
    const u8 *text, const int len,
    const int start, const int stop,
    float confab_bias, int confab_size, int confab_line_end,
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
  float report_scale = 1.0f / ((model->report_interval - report_counter) * n_nets);

  struct timespec timers[2];
  struct timespec *time_start = timers;
  struct timespec *time_end = timers + 1;
  clock_gettime(CLOCK_MONOTONIC, time_start);
  for(i = start; i < len - 1; i++){
    float momentum = rnn_calculate_momentum_soft_start(net->generation,
        model->momentum, model->momentum_soft_start);
    if (n_nets > 1 || model->learning_style != RNN_MOMENTUM_WEIGHTED ||
        model->use_multi_tap_path){
      for (j = 0; j < n_nets; j++){
        RecurNN *n = nets[j];
        int offset = i + j * spacing;
        if (offset >= len - 1){
          offset -= len - 1;
        }
        rnn_bptt_advance(n);
        e = net_error_bptt(n, n->bptt->o_error,
            text[offset], text[offset + 1], &c);
        correct += c;
        error += e;
        entropy += capped_log2f(1.0f - e);
        /*Second argument to r_b_c_deltas toggles delta accumulation. Turning
          it off on the first run avoids expicit zeroing outside of the loop
          (via rnn_bptt_clear_deltas) and is thus slightly faster.
         */
        rnn_bptt_calc_deltas(n, j ? 1 : 0);
      }
      rnn_apply_learning(net, model->learning_style, momentum);
    }
    else {
      RecurNNBPTT *bptt = net->bptt;
      bptt->momentum = momentum;
      rnn_bptt_advance(net);
      e = net_error_bptt(net, bptt->o_error, text[i], text[i + 1], &c);
      rnn_bptt_calculate(net, model->batch_size);
      correct += c;
      error += e;
      entropy += capped_log2f(1.0f - e);
    }

    if (model->images.input_ppm){
      temporal_ppm_add_row(model->images.input_ppm, net->input_layer);
    }
    if (model->images.error_ppm){
      temporal_ppm_add_row(model->images.error_ppm, net->bptt->o_error);
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
        int k = net->generation >> 10;
        entropy *= -report_scale;
        error *= report_scale;
        float accuracy = correct * report_scale;
        double per_sec = 1.0 / report_scale / elapsed;
        if (confab_net && confab_size && quietness < 1){
          if (confab_line_end >= 0){
            fprintf(stderr, C_GREY "%5dk t%.2f " C_RED "v%.2f"
                C_NORMAL " %.0f/s |" C_NORMAL,
                k, entropy, ventropy, per_sec + 0.5);
            rnn_char_fconfab_variable(stderr, confab_net,
                confab_line_end, confab_size, model->alphabet, confab_bias);
          }
          else{
            fprintf(stderr, C_GREY "%5dk e.%02d t%.2f v%.2f a.%02d %.0f/s |" C_NORMAL,
                k, (int)(error * 100 + 0.5),
                entropy, ventropy,
                (int)(accuracy * 100 + 0.5), per_sec + 0.5);
            int alloc_size = confab_size * 4;
            char confab[alloc_size + 1];
            rnn_char_confabulate(confab_net, confab, confab_size, alloc_size,
                model->alphabet, confab_bias, -1);
            STDERR_DEBUG("%s|", confab);
          }
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
        report_scale = 1.0f / (model->report_interval * n_nets);
      }

      if (model->save_net && model->filename){
        rnn_save_net(net, model->filename, 1);
      }
      if (model->images.periodic_pgm_dump_string){
        rnn_multi_pgm_dump(net, model->images.periodic_pgm_dump_string,
            model->images.basename);
      }
      model->schedule.eval(model, ventropy, quietness < 2);
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


double
rnn_char_cross_entropy(RecurNN *net, RnnCharAlphabet *alphabet,
    const u8 *text, const int len, const int skip,
    const u8 *prefix_text, const int prefix_len){
  if (prefix_text){
    for(int i = 0; i < prefix_len; i++){
      one_hot_opinion(net, prefix_text[i], 0);
    }
  }
  double entropy = get_cross_entropy(net, text, len, skip);
  return entropy;
}
