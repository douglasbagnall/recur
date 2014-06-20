#ifndef HAVE_CHAR_MODEL_H
#define HAVE_CHAR_MODEL_H
#include "recur-nn.h"

#include <ctype.h>
#include <stdbool.h>

typedef struct Schedule_ Schedule;

struct Schedule_ {
  float *recent;
  int recent_len;
  int timeout;
  float learn_rate_mul;
  float learn_rate_min;
  void (*eval)(Schedule *s, RecurNN *net, float score, int verbose);
};

typedef struct _Ventropy {
  RecurNN *net;
  int counter;
  float *history;
  const u8 *text;
  int len;
  int lap;
  int lapsize;
  float entropy;
} Ventropy;

typedef struct _RnnCharModel RnnCharModel;

struct _RnnCharModel {
  RecurNN *net;
  RecurNN **training_nets;
  int n_training_nets;

  char *pgm_name;
  uint batch_size;

  char *filename;
  float momentum;
  float momentum_soft_start;
  int momentum_style;
  bool learn_caps;
  bool periodic_pgm_dump;
  bool temporal_pgm_dump;
  float periodic_weight_noise;
  int quiet;
  uint report_interval;
  bool save_net;
  bool use_multi_tap_path;

  TemporalPPM *input_ppm;
  TemporalPPM *error_ppm;
  char *alphabet;
  char * periodic_pgm_dump_string;
  Schedule schedule;
};

struct CharMetadata {
  char *alphabet;
  char *collapse_chars;
  int learn_caps;
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

void init_schedule(Schedule *s, int recent_len,
    float learn_rate_min, float learn_rate_mul);

float rnn_char_calc_ventropy(RnnCharModel *model, Ventropy *v, int lap);

void confabulate(RecurNN *net, char *text, int len, const char* alphabet,
    float bias, int learn_caps);

void init_ventropy(Ventropy *v, RecurNN *net, const u8 *text,
    const int len, const int lap);

int epoch(RnnCharModel *model, RecurNN *confab_net, Ventropy *v,
    const u8 *text, const int len,
    const int start, const int stop,
    float confab_bias, int confab_size,
    int quietness);

char *rnn_char_construct_metadata(const struct CharMetadata *m);
int rnn_char_load_metadata(const char *metadata, struct CharMetadata *m);

void rnn_char_free_metadata_items(struct CharMetadata *m);

char* construct_net_filename(struct CharMetadata *m,
    const char *basename, int bottom_size, int hidden_size, int learn_caps);

#endif
