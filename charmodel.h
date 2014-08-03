#ifndef HAVE_CHAR_MODEL_H
#define HAVE_CHAR_MODEL_H
#include "recur-nn.h"
#include "pgm_dump.h"

#include <ctype.h>
#include <stdbool.h>

enum {
  RNN_CHAR_FLAG_CASE_INSENSITIVE = 1,
  RNN_CHAR_FLAG_UTF8 = 2,
  RNN_CHAR_FLAG_COLLAPSE_SPACE = 4
} rnn_char_flags;

typedef struct _RnnCharSchedule RnnCharSchedule;
typedef struct _RnnCharModel RnnCharModel;

typedef struct _RnnCharImageSettings{
  char *basename;
  bool periodic_pgm_dump;
  bool temporal_pgm_dump;
  TemporalPPM *input_ppm;
  TemporalPPM *error_ppm;
  char * periodic_pgm_dump_string;
} RnnCharImageSettings;

struct _RnnCharSchedule {
  float *recent;
  int recent_len;
  int timeout;
  float learn_rate_mul;
  float learn_rate_min;
  int adjust_noise;
  void (*eval)(RnnCharModel *m, float score, int verbose);
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


struct _RnnCharModel {
  RecurNN *net;
  RecurNN **training_nets;
  int n_training_nets;
  uint batch_size;
  char *filename;
  float momentum;
  float momentum_soft_start;
  int momentum_style;
  float periodic_weight_noise;
  uint report_interval;
  bool save_net;
  bool use_multi_tap_path;
  u32 flags;
  int *alphabet; /*unicode points */
  int *collapse_chars;
  RnnCharSchedule schedule;
  RnnCharImageSettings images;
};

typedef struct RnnCharMetadata {
  char *alphabet; /*utf-8 or byte string */
  char *collapse_chars;
  bool utf8;
} RnnCharMetadata;

typedef struct _RnnCharClassifiedChar{
  u8 class;
  u8 symbol;
} RnnCharClassifiedChar;

typedef struct _RnnCharClassifiedText{
  RnnCharClassifiedChar *text;
  int len;
  int *alphabet;
  int a_len;
  int *collapse_chars;
  int c_len;
  u32 flags;
  int lag;
  int n_classes;
  char **classes;
  bool utf8;
} RnnCharClassifiedText;

typedef struct _RnnCharClassifier{
  RnnCharClassifiedText *text;
  RecurNN *net;
  RecurNN **training_nets;
  int n_training_nets;
  char *pgm_name;
  uint batch_size;
  char *filename;
  float momentum;
  float momentum_soft_start;
  int momentum_style;
  float periodic_weight_noise;
  uint report_interval;
  bool save_net;
  RnnCharSchedule schedule;
  RnnCharImageSettings images;
} RnnCharClassifier;




typedef struct _RnnCharClassBlock RnnCharClassBlock;
struct _RnnCharClassBlock
{
  const char *class_name;
  const char *text;
  RnnCharClassBlock *next;
  int len;
  u8 class_code;
};

#define NO_CLASS 0xFF

int rnn_char_classify_epoch(RnnCharClassifier *model);

int rnn_char_alloc_file_contents(const char *filename, char **contents, int *len);

RnnCharClassifiedChar *
rnn_char_alloc_classified_text(RnnCharClassBlock *b,
    int *alphabet, int a_len, int *collapse_chars, int c_len,
    int *text_len, u32 flags);

void rnn_char_adjust_text_lag(RnnCharClassifiedText *t, int lag);

int rnn_char_find_alphabet_s(const char *text, int len, int *alphabet, int *a_len,
    int *collapse_chars, int *c_len, double threshold, double digit_adjust,
    double alpha_adjust, u32 flags);

int rnn_char_find_alphabet_f(const char *filename, int *alphabet, int *a_len,
    int *collapse_chars, int *c_len, double threshold, double digit_adjust,
    double alpha_adjust, u32 flags);

u8* rnn_char_alloc_collapsed_text(char *filename, int *alphabet, int a_len,
    int *collapse_chars, int c_len, int *text_len, u32 flags, int quietness);

void rnn_char_dump_collapsed_text(const u8 *text, int len, const char *name,
    const char *alphabet);

void rnn_char_init_schedule(RnnCharSchedule *s, int recent_len,
    float learn_rate_min, float learn_rate_mul, int adjust_noise);

float rnn_char_calc_ventropy(RnnCharModel *model, RnnCharVentropy *v, int lap);

int rnn_char_confabulate(RecurNN *net, char *dest, int char_len,
    int byte_len, const int* alphabet, int utf8, float bias);

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
    const char *basename, int input_size, int bottom_size, int hidden_size,
    int output_size);

void
rnn_char_dump_alphabet(int *alphabet, int len, int utf8);

#endif
