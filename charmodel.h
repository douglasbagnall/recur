#ifndef HAVE_CHAR_MODEL_H
#define HAVE_CHAR_MODEL_H
#include "recur-nn.h"

#include <ctype.h>
#include <stdbool.h>

typedef struct _RnnCharSchedule RnnCharSchedule;

struct _RnnCharSchedule {
  float *recent;
  int recent_len;
  int timeout;
  float learn_rate_mul;
  float learn_rate_min;
  void (*eval)(RnnCharSchedule *s, RecurNN *net, float score, int verbose);
};

typedef struct _RnnCharVentropy {
  RecurNN *net;
  int counter;
  float *history;
  const u8 *text;
  int len;
  int lap;
  int lapsize;
  float entropy;
} RnnCharVentropy;

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
  RnnCharSchedule schedule;
};

struct RnnCharMetadata {
  char *alphabet;
  char *collapse_chars;
  int learn_caps;
};


u8* rnn_char_alloc_collapsed_text(char *filename, const char *alphabet,
    const u8 *collapse_chars, long *len, int learn_caps, int quietness);

void rnn_char_dump_collapsed_text(const u8 *text, int len, const char *name,
    const char *alphabet);

void rnn_char_init_schedule(RnnCharSchedule *s, int recent_len,
    float learn_rate_min, float learn_rate_mul);

float rnn_char_calc_ventropy(RnnCharModel *model, RnnCharVentropy *v, int lap);

void rnn_char_confabulate(RecurNN *net, char *text, int len, const char* alphabet,
    float bias, int learn_caps);

void rnn_char_init_ventropy(RnnCharVentropy *v, RecurNN *net, const u8 *text,
    const int len, const int lap);

int rnn_char_epoch(RnnCharModel *model, RecurNN *confab_net, RnnCharVentropy *v,
    const u8 *text, const int len,
    const int start, const int stop,
    float confab_bias, int confab_size,
    int quietness);

char *rnn_char_construct_metadata(const struct RnnCharMetadata *m);
int rnn_char_load_metadata(const char *metadata, struct RnnCharMetadata *m);

void rnn_char_free_metadata_items(struct RnnCharMetadata *m);

char* rnn_char_construct_net_filename(struct RnnCharMetadata *m,
    const char *basename, int bottom_size, int hidden_size, int learn_caps);

#endif
