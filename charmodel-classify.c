/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL

Classify text, possibly by language or author.
*/

#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include <math.h>
#include "path.h"
#include "badmaths.h"
#include "colour.h"
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

/*Adjust the lag of the text predictions in-place. Anything that gets adjusted
  out of range gets its target set to 0xff, which indicates no training */

void
rnn_char_adjust_text_lag(RnnCharClassifiedText *t, int lag){
  int i;
  RnnCharClassifiedChar *text = t->text;
  int len = t->len;
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
  t->lag += lag;
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
rnn_char_classify_epoch(RnnCharClassifier *model){
  int n_nets = model->n_training_nets;
  RnnCharClassifiedText *t = model->text;
  int len = t->len;
  RnnCharClassifiedChar *text = t->text;
  int spacing = len / n_nets;
  int correct = 0;
  float mean_error = 0;
  float t_entropy = 0;
  int examples_seen = 0;

  RecurNN *net = model->net;
  RecurNN **nets = model->training_nets;
  uint report_counter = net->generation % model->report_interval;
  struct timespec timers[2];
  struct timespec *time_start = timers;
  struct timespec *time_end = timers + 1;

  RecurNN *vnet = NULL;
  if (t->validation_len){
    vnet = rnn_clone(net,
        net->flags & ~(RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS),
        RECUR_RNG_SUBSEED, NULL);
  }

  /*roll through a few before training begins, to prime the net */
  int prime = MIN(len / (n_nets * 20), 10);
  for (int i = 0; i < prime; i++){
    int offset = i;
    for (int j = 0; j < n_nets; j++){
      RnnCharClassifiedChar cc = text[offset];
      RecurNN *n = nets[j];
      one_hot_opinion(n, cc.symbol, net->presynaptic_noise);
      offset += spacing;
      if (offset >= len){
        offset -= len;
      }
    }
  }

  for (int i = prime; i < len; i++){
    float momentum = rnn_calculate_momentum_soft_start(net->generation,
        model->momentum, model->momentum_soft_start);
    int offset = i;
    for (int j = 0; j < n_nets; j++){
      RnnCharClassifiedChar cc = text[offset];
      RecurNN *n = nets[j];
      rnn_bptt_advance(n);
      int class = cc.class;
      MAYBE_DEBUG("j %i offset %i symbol %i class %i output_size %i len %i",
          j, offset, cc.symbol, class, n->output_size, len);
      float *answer = one_hot_opinion(n, cc.symbol, net->presynaptic_noise);
      if (class != NO_CLASS){
        float *error = n->bptt->o_error;
        ASSUME_ALIGNED(error);
        int winner = softmax_best_guess(error, answer, n->output_size);
        correct += (winner == class);
        float e = error[class] + 1.0f;
        error[class] = e;
        //rnn_bptt_calculate(n, model->batch_size);
        mean_error += e;
        t_entropy -= capped_log2f(1.0f - e);
        examples_seen++;
        if (n == net){
          rnn_log_int(net, "skipping", 0);
        }
        //DEBUG("winner %i correct %i error %.2f", winner, winner == class, e);
        rnn_bptt_calc_deltas(n, j ? 1 : 0);
      }
      else {
        if (n == net){
          rnn_log_int(net, "skipping", 1);
        }
      }

      offset += spacing;
      if (offset >= len){
        offset -= len;
      }
    }
    rnn_apply_learning(net, model->learning_style, momentum);

    report_counter++;
    if (report_counter == model->report_interval){
      report_counter = 0;
      double elapsed = get_elapsed_interval(&time_start, &time_end);
      float scale = 1.0f / examples_seen;
      t_entropy *= scale;
      mean_error *= scale;
      float accuracy = correct * scale;
      double per_sec = examples_seen / elapsed;
      rnn_log_float(net, "t_error", mean_error);
      rnn_log_float(net, "t_entropy", t_entropy);
      rnn_log_float(net, "momentum", net->bptt->momentum);
      rnn_log_float(net, "accuracy", accuracy);
      rnn_log_float(net, "learn-rate", net->bptt->learn_rate);
      rnn_log_float(net, "per_second", per_sec);
      float v_entropy = 0;
      float v_error = 0;

      if (vnet){
        float error[vnet->output_size];
        int vlen = t->validation_len;
        RnnCharClassifiedChar *vtext = t->validation_text;
        int div = 0;
        for (int j = 0; j < vlen; j++){
          RnnCharClassifiedChar cc = vtext[j];
          if (cc.class != NO_CLASS){
            float *answer = one_hot_opinion(vnet, cc.symbol, 0);
            softmax(error, answer, net->output_size);
            float e = error[cc.class];
            v_error += 1.0 - e;
            v_entropy -= capped_log2f(e);
            MAYBE_DEBUG("class %d symbol %d error %.3g entropy %.3g",
                cc.class, cc.symbol, e, capped_log2f(e));
            div++;
          }
        }
        v_entropy /= div;
        v_error /= div;
        rnn_log_float(net, "v_entropy", v_entropy);
        rnn_log_float(net, "v_error", v_error);
      }

      for (int j = 0; j < net->output_size; j++){
        float x = -net->bptt->o_error[j];
        int is_target = x < 0.0;
        if (is_target){
          fprintf(stderr, C_RED);
          x += 1.0;
        }
        int c = x * 9.0 + 0.5;
        if (c == 0){
          fprintf(stderr, is_target ? "." C_NORMAL : " ");
        }
        else {
          fprintf(stderr, "\xe2\x96%c" C_NORMAL, 128 + c);
        }
      }

      DEBUG(" v_entropy %.2f v_error %.2f t_entropy %.2f acc. %.2f error %.2f "
          "speed %.1f (%d examples)",
          v_entropy, v_error, t_entropy, accuracy, mean_error, per_sec, examples_seen);

      correct = 0;
      mean_error = 0.0f;
      t_entropy = 0.0f;
      examples_seen = 0;

      if (model->save_net && model->filename){
        rnn_save_net(net, model->filename, 1);
      }
    }
  }
  return 0;
}
