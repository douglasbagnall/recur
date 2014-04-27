#include "recur-nn.h"

typedef struct Schedule_ Schedule;

struct Schedule_ {
  float *recent;
  int recent_len;
  float margin;
  int timeout;
  float learn_rate_mul;
  float learn_rate_min;
  void (*eval)(Schedule *s, RecurNN *net, float score, int verbose);
};


static void
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

static void
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
