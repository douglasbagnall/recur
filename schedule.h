#include "recur-nn.h"

typedef struct Schedule_ Schedule;

struct Schedule_ {
  float *history;
  int history_len;
  int history_i;
  float margin;
  float learn_rate_mul;
  float learn_rate_min;
  void (*eval)(Schedule *s, RecurNN *net, float score);
};


static void
eval_simple(Schedule *s, RecurNN *net, float score){
  float sum = 0.0f;
  int i;
  RecurNNBPTT *bptt = net->bptt;
  if (bptt->learn_rate <= s->learn_rate_min){
    return;
  }
  for (i = 0; i < s->history_len; i++){
    sum += s->history[i];
  }
  if (sum < (score + s->margin) * s->history_len){
    bptt->learn_rate = MAX(s->learn_rate_min, bptt->learn_rate * s->learn_rate_mul);
    DEBUG("sum %g, score %g margin %g;setting learn_rate to %g",
        sum, score, s->margin, bptt->learn_rate);
    /*postpone next update for at least a history cycle*/
    score = 1e10;
  }
  s->history[s->history_i] = score;
  s->history_i++;
  if (s->history_i >= s->history_len){
    s->history_i = 0;
  }
}

static void
init_schedule(Schedule *s, int history_len, float margin,
    float learn_rate_min, float learn_rate_mul){
  s->history = malloc_aligned_or_die(history_len * sizeof(float));
  s->history_len = history_len;
  s->history_i = 0;
  s->learn_rate_min = learn_rate_min;
  s->learn_rate_mul = learn_rate_mul;
  s->margin = margin;
  for (int i = 0; i < s->history_len; i++){
    s->history[i] = 1e20;
  }
  s->eval = eval_simple;
}
