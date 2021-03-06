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
  bool temporal_pgm_dump;
  TemporalPPM *input_ppm;
  TemporalPPM *error_ppm;
  char *periodic_pgm_dump_string;
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

typedef struct _RnnCharAlphabet {
  int *points;
  int *collapsed_points;
  int len;
  int collapsed_len;
  u32 flags;
} RnnCharAlphabet;


struct _RnnCharModel {
  RecurNN *net;
  RecurNN **training_nets;
  int n_training_nets;
  uint batch_size;
  char *filename;
  float momentum;
  float momentum_soft_start;
  int learning_style;
  float periodic_weight_noise;
  uint report_interval;
  bool save_net;
  bool use_multi_tap_path;
  RnnCharAlphabet *alphabet;
  RnnCharSchedule schedule;
  RnnCharImageSettings images;
};

typedef struct RnnCharMetadata {
  char *alphabet; /*utf-8 or byte string */
  char *collapse_chars;
  bool utf8;
  bool case_insensitive;
  bool collapse_space;
} RnnCharMetadata;

typedef struct _RnnCharClassifiedChar{
  u8 class;
  u8 symbol;
} RnnCharClassifiedChar;

typedef struct _RnnCharClassifiedText{
  RnnCharClassifiedChar *text;
  int len;
  RnnCharClassifiedChar *validation_text;
  int validation_len;
  RnnCharAlphabet *alphabet;
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
  int learning_style;
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


typedef struct _RnnCharProgressReport
{
  float training_entropy;
  float training_error;
  float training_accuracy;
  float per_second;
} RnnCharProgressReport;

typedef struct RnnCharMultiConfab
{
  RecurNN **nets;
  int *last_char;
  int caps_marker;
  char **strings;
  uint n_classes;
  uint char_len;
  uint byte_len;
  float bias;
  uint period;
  RnnCharAlphabet *alphabet;
} RnnCharMultiConfab;


#define NO_CLASS 0xFF

int rnn_char_classify_epoch(RnnCharClassifier *model);

int rnn_char_alloc_file_contents(const char *filename, char **contents, int *len);

RnnCharClassifiedChar * rnn_char_alloc_classified_text(RnnCharClassBlock *b,
    RnnCharAlphabet *alphabet, int *text_len, int ignore_start);

void rnn_char_adjust_text_lag(RnnCharClassifiedText *t, int lag);

int rnn_char_find_alphabet_s(const char *text, int len, RnnCharAlphabet *alphabet,
    double threshold, double digit_adjust, double alpha_adjust);

int rnn_char_find_alphabet_f(const char *filename, RnnCharAlphabet *alphabet,
    double threshold, double digit_adjust, double alpha_adjust);

u8 *rnn_char_alloc_encoded_text(RnnCharAlphabet *alphabet, const char *text,
    int byte_len, int *encoded_len, int *char_to_net, bool verbose);

u8* rnn_char_load_new_encoded_text(const char *filename,
    RnnCharAlphabet *alphabet, int *encoded_len, int quietness);

char *rnn_char_uncollapse_text(RnnCharAlphabet *alphabet, const u8 *orig, int len,
    int *dest_len);

void rnn_char_dump_collapsed_text(const u8 *text, int len, const char *name,
    const char *alphabet);

void rnn_char_init_schedule(RnnCharSchedule *s, int recent_len,
    float learn_rate_min, float learn_rate_mul, int adjust_noise);

float rnn_char_calc_ventropy(RnnCharModel *model, RnnCharVentropy *v, int lap);

int rnn_char_confabulate(RecurNN *net, char *dest, int char_len,
    int byte_len, RnnCharAlphabet* a, float bias, int *prev_char,
    int start_point, int stop_point);

void rnn_char_delete_ventropy(RnnCharVentropy *v);

void rnn_char_init_ventropy(RnnCharVentropy *v, RecurNN *net, const u8 *text,
    const int len, const int lap);

int rnn_char_epoch(RnnCharModel *model, RecurNN *confab_net, RnnCharVentropy *v,
    const u8 *text, const int len,
    const int start, const int stop,
    float confab_bias, int confab_size, int confab_line_end,
    int quietness, uint diagonal_only_section, uint diagonal_only_friends);

char *rnn_char_construct_metadata(const struct RnnCharMetadata *m);
int rnn_char_load_metadata(const char *metadata, struct RnnCharMetadata *m);

void rnn_char_free_metadata_items(struct RnnCharMetadata *m);

char* rnn_char_construct_net_filename(struct RnnCharMetadata *m,
    const char *basename, int input_size, int bottom_size, int hidden_size,
    int output_size);

int rnn_char_check_metadata(RecurNN *net, struct RnnCharMetadata *m,
    bool trust_file_metadata, bool force_metadata);

void rnn_char_copy_metadata_items(struct RnnCharMetadata *src,\
    struct RnnCharMetadata *dest);

RnnCharAlphabet *rnn_char_new_alphabet(void);

void rnn_char_dump_alphabet(RnnCharAlphabet *alphabet);

void rnn_char_free_alphabet(RnnCharAlphabet *a);

int rnn_char_get_codepoint(RnnCharAlphabet *a, const char *s);

RnnCharAlphabet *rnn_char_new_alphabet_from_net(RecurNN *net);

int *rnn_char_new_char_lut(const RnnCharAlphabet *alphabet);

void rnn_char_alphabet_set_flags(RnnCharAlphabet *a,
    bool case_insensitive, bool utf8, bool collapse_space);

int rnn_char_prime(RecurNN *net, RnnCharAlphabet *alphabet,
    const u8 *text, const int len);

double rnn_char_cross_entropy(RecurNN *net, RnnCharAlphabet *alphabet,
    const u8 *text, const int len, const int ignore_first,
    const u8 *prefix_text, const int prefix_len);


void
rnn_char_multitext_train(RecurNN *net, u8 *text, int len, int alphabet_len,
    int target_class, float leakage, RnnCharProgressReport *report,
    RnnCharMultiConfab *confab,
    int learning_style, float momentum, int batch_size,
    TemporalPPM *input_ppm, TemporalPPM *error_ppm,
    const char *periodic_pgm_string, int periodic_pgm_period);


void rnn_char_multitext_spin(RecurNN *net, u8 *text, int len,
    TemporalPPM *input_ppm, TemporalPPM *error_ppm,
    const char *periodic_pgm_string, int periodic_pgm_period);

void
rnn_char_multi_cross_entropy(RecurNN *net, const u8 *text, int len,
    int alphabet_len, double *entropy, int ignore_start);

RnnCharMultiConfab *
rnn_char_new_multi_confab(RecurNN *net, RnnCharAlphabet *alhapbet, int n_classes,
    int target_len, uint confab_period, int caps_marker);


void
rnn_char_free_multi_confab(RnnCharMultiConfab *mc);


#endif
