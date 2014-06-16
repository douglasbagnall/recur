#ifndef HAVE_CHAR_MODEL_H
#define HAVE_CHAR_MODEL_H

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

u8* alloc_and_collapse_text(char *filename, const char *alphabet,
    const u8 *collapse_chars, long *len, int learn_caps, int quietness);

void dump_collapsed_text(const u8 *text, int len, const char *name,
    const char *alphabet);

float net_error_bptt(RecurNN *net, float *restrict error, int c, int next,
    int *correct, int learn_caps);

int opinion_deterministic(RecurNN *net, int hot, int learn_caps);
int opinion_probabilistic(RecurNN *net, int hot, float bias, int learn_caps);

float validate(RecurNN *net, const u8 *text, int len, int learn_caps);


static inline float
capped_log2f(float x){
  return (x < 1e-30f) ? -100.0f : log2f(x);
}

void init_schedule(Schedule *s, int recent_len, float margin,
    float learn_rate_min, float learn_rate_mul);

void eval_simple(Schedule *s, RecurNN *net, float score, int verbose);

#endif
