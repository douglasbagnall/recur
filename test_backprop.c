#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include <math.h>
#include "path.h"
#include "badmaths.h"
#include "schedule.h"
#include "ccan/opt/opt.h"
#include <errno.h>
#include <stdio.h>
#include <fenv.h>
#include <ctype.h>

#define DICKENS_SHUFFLED_TEXT TEST_DATA_DIR "/dickens-shuffled.txt"
#define DICKENS_TEXT TEST_DATA_DIR "/dickens.txt"
#define EREWHON_TEXT TEST_DATA_DIR "/erewhon.txt"
#define EREWHON_LONG_TEXT TEST_DATA_DIR "/erewhon-erewhon"\
  "-revisited-sans-gutenberg.txt"

#define CONFAB_SIZE 80

#define DEFAULT_PERIODIC_PGM_DUMP 0
#define DEFAULT_TEMPORAL_PGM_DUMP 0
#define DEFAULT_RELOAD 0
#define DEFAULT_LEARN_RATE 0.001
#define DEFAULT_LEARN_RATE_MIN DEFAULT_LEARN_RATE
#define DEFAULT_LEARN_RATE_INERTIA 60
#define DEFAULT_LEARN_RATE_SCALE 0.4
#define DEFAULT_BPTT_DEPTH 30
#define DEFAULT_BPTT_ADAPTIVE_MIN 1
#define DEFAULT_MOMENTUM 0.95
#define DEFAULT_MOMENTUM_WEIGHT RNN_MOMENTUM_WEIGHT
#define DEFAULT_MOMENTUM_SOFT_START 0
#define DEFAULT_BIAS 1
#define DEFAULT_RNG_SEED 1
#define DEFAULT_STOP 0
#define DEFAULT_BATCH_SIZE 1
#define DEFAULT_VALIDATE_CHARS 0
#define DEFAULT_VALIDATION_OVERLAP 1
#define DEFAULT_DROPOUT 0
#define DEFAULT_OVERRIDE 0
#define DEFAULT_DETERMINISTIC_CONFAB 0
#define DEFAULT_SAVE_NET 1
#define DEFAULT_LOG_FILE "bptt.log"
#define DEFAULT_START_CHAR -1
#define DEFAULT_HIDDEN_SIZE 199
#define DEFAULT_DENSE_WEIGHTS 0
#define DEFAULT_PERFORATE_WEIGHTS 0.0f
#define DEFAULT_LEARN_CAPITALS 0
#define DEFAULT_DUMP_COLLAPSED_TEXT NULL
#define DEFAULT_MULTI_TAP 0
#define DEFAULT_USE_MULTI_TAP_PATH 0
#define DEFAULT_MOMENTUM_STYLE RNN_MOMENTUM_WEIGHTED
#define DEFAULT_WEIGHT_SCALE_FACTOR 0
#define DEFAULT_REPORT_INTERVAL 1024
#define DEFAULT_CONFAB_ONLY 0
#define DEFAULT_DIAGONAL_BOOST 0

#define BELOW_QUIET_LEVEL(quiet) if (opt_quiet < quiet)

#define Q_DEBUG(quiet, ...) do {                               \
                                if (opt_quiet < quiet)         \
                                  STDERR_DEBUG(__VA_ARGS__); \
                                } while(0)


#define NET_TO_CHAR "#abcdefghijklmnopqrstuvwxyz,'- .\";:!?"
#define HASH_CHARS "1234567890&@"


static TemporalPPM *input_ppm;
static TemporalPPM *error_ppm;

static uint opt_hidden_size = DEFAULT_HIDDEN_SIZE;
static uint opt_bptt_depth = DEFAULT_BPTT_DEPTH;
static float opt_learn_rate = DEFAULT_LEARN_RATE;
static float opt_learn_rate_min = DEFAULT_LEARN_RATE_MIN;
static int  opt_learn_rate_inertia = DEFAULT_LEARN_RATE_INERTIA;
static float opt_learn_rate_scale = DEFAULT_LEARN_RATE_SCALE;
static float opt_momentum = DEFAULT_MOMENTUM;
static int opt_quiet = 0;
static char * opt_filename = NULL;
static char * opt_logfile = DEFAULT_LOG_FILE;
static char * opt_alphabet = NET_TO_CHAR;
static char * opt_collapse_chars = HASH_CHARS;
static char * opt_textfile = DICKENS_SHUFFLED_TEXT;
static char * opt_dump_collapsed_text = DEFAULT_DUMP_COLLAPSED_TEXT;
static bool opt_bias = DEFAULT_BIAS;
static bool opt_reload = DEFAULT_RELOAD;
static float opt_dropout = DEFAULT_DROPOUT;
static float opt_momentum_weight = DEFAULT_MOMENTUM_WEIGHT;
static float opt_momentum_soft_start = DEFAULT_MOMENTUM_SOFT_START;
static u64 opt_rng_seed = DEFAULT_RNG_SEED;
static int opt_stop = DEFAULT_STOP;
static int opt_validate_chars = DEFAULT_VALIDATE_CHARS;
static int opt_validation_overlap = DEFAULT_VALIDATION_OVERLAP;
static int opt_start_char = DEFAULT_START_CHAR;
static bool opt_override = DEFAULT_OVERRIDE;
static bool opt_bptt_adaptive_min = DEFAULT_BPTT_ADAPTIVE_MIN;
static uint opt_batch_size = DEFAULT_BATCH_SIZE;
static uint opt_dense_weights = DEFAULT_DENSE_WEIGHTS;
static float opt_perforate_weights = DEFAULT_PERFORATE_WEIGHTS;
static bool opt_temporal_pgm_dump = DEFAULT_TEMPORAL_PGM_DUMP;
static bool opt_periodic_pgm_dump = DEFAULT_PERIODIC_PGM_DUMP;
static bool opt_deterministic_confab = DEFAULT_DETERMINISTIC_CONFAB;
static bool opt_save_net = DEFAULT_SAVE_NET;
static bool opt_learn_capitals = DEFAULT_LEARN_CAPITALS;
static uint opt_multi_tap = DEFAULT_MULTI_TAP;
static bool opt_use_multi_tap_path = DEFAULT_USE_MULTI_TAP_PATH;
static int opt_momentum_style = DEFAULT_MOMENTUM_STYLE;
static float opt_weight_scale_factor = DEFAULT_WEIGHT_SCALE_FACTOR;
static uint opt_report_interval = DEFAULT_REPORT_INTERVAL;
static uint opt_confab_only = DEFAULT_CONFAB_ONLY;
static float opt_diagonal_boost = DEFAULT_DIAGONAL_BOOST;


/* Following ccan/opt/helpers.c opt_set_longval, etc */
static char *
opt_set_floatval(const char *arg, float *f)
{
  char *endp;
  errno = 0;
  *f = strtof(arg, &endp);
  if (*endp || !arg[0] || errno){
    char *s;
    if (asprintf(&s, "'%s' doesn't seem like a number", arg) > 0){
      return s;
    }
  }
  return NULL;
}

void opt_show_floatval(char buf[OPT_SHOW_LEN], const float *f)
{
  snprintf(buf, OPT_SHOW_LEN, "%g", *f);
}

static struct opt_table options[] = {
  OPT_WITH_ARG("-H|--hidden-size=<n>", opt_set_uintval, opt_show_uintval,
      &opt_hidden_size, "number of hidden nodes"),
  OPT_WITH_ARG("-d|--depth=<n>", opt_set_uintval, opt_show_uintval,
      &opt_bptt_depth, "max depth of BPTT recursion"),
  OPT_WITH_ARG("-r|--rng-seed=<seed>", opt_set_ulongval_bi, opt_show_ulongval_bi,
      &opt_rng_seed, "RNG seed (-1 for auto)"),
  OPT_WITH_ARG("-s|--stop-after=<n>", opt_set_intval_bi, opt_show_intval_bi,
      &opt_stop, "Stop at generation n (0: no stop, negative means relative)"),
  OPT_WITH_ARG("--batch-size=<n>", opt_set_uintval_bi, opt_show_uintval_bi,
      &opt_batch_size, "bptt minibatch size"),
  OPT_WITH_ARG("--dense-weights=<n>", opt_set_uintval_bi, opt_show_uintval_bi,
      &opt_dense_weights, "no initial zero weights; > 1 for many near zero"),
  OPT_WITH_ARG("--perforate-weights=<0-1>", opt_set_floatval, opt_show_floatval,
      &opt_perforate_weights, "Zero this portion of weights"),
  OPT_WITH_ARG("-V|--validate-chars=<n>", opt_set_intval_bi, opt_show_intval_bi,
      &opt_validate_chars, "Retain this many characters for validation"),
  OPT_WITH_ARG("--validation-overlap=<n>", opt_set_intval, opt_show_intval,
      &opt_validation_overlap, "> 1 to use lapped validation (quicker)"),
  OPT_WITH_ARG("--start-char=<n>", opt_set_intval, opt_show_intval,
      &opt_start_char, "character to start epoch on (-1 for auto)"),
  OPT_WITH_ARG("-l|--learn-rate=<0-1>", opt_set_floatval, opt_show_floatval,
      &opt_learn_rate, "initial learning rate"),
  OPT_WITH_ARG("--learn-rate-min=<0-1>", opt_set_floatval, opt_show_floatval,
      &opt_learn_rate_min, "minimum learning rate (>learn-rate is off)"),
  OPT_WITH_ARG("--learn-rate-inertia=<int>", opt_set_intval, opt_show_intval,
      &opt_learn_rate_inertia, "tardiness of learn-rate reduction (try 30-90)"),
  OPT_WITH_ARG("--learn-rate-scale=<int>", opt_set_floatval, opt_show_floatval,
      &opt_learn_rate_scale, "size of learn rate reductions"),
  OPT_WITH_ARG("-m|--momentum=<float>", opt_set_floatval, opt_show_floatval,
      &opt_momentum, "momentum"),
  OPT_WITH_ARG("--momentum-weight=<float>", opt_set_floatval, opt_show_floatval,
      &opt_momentum_weight, "momentum weight"),
  OPT_WITH_ARG("--momentum-soft-start=<float>", opt_set_floatval, opt_show_floatval,
      &opt_momentum_soft_start, "softness of momentum onset (0 for constant)"),
  OPT_WITHOUT_ARG("-q|--quiet", opt_inc_intval,
      &opt_quiet, "print less (twice for even less)"),
  OPT_WITHOUT_ARG("-v|--verbose", opt_dec_intval,
      &opt_quiet, "print more, if possible"),
  OPT_WITHOUT_ARG("--bias", opt_set_bool,
      &opt_bias, "use bias (default)"),
  OPT_WITHOUT_ARG("--no-bias", opt_set_invbool,
      &opt_bias, "Don't use bias"),
  OPT_WITHOUT_ARG("-R|--reload", opt_set_bool,
      &opt_reload, "try to reload the net"),
  OPT_WITHOUT_ARG("-N|--no-reload", opt_set_invbool,
      &opt_reload, "Don't try to reload"),
  OPT_WITHOUT_ARG("--bptt-adaptive-min", opt_set_bool,
      &opt_bptt_adaptive_min, "auto-adapt BPTT minimum error threshold (default)"),
  OPT_WITHOUT_ARG("--no-bptt-adaptive-min", opt_set_invbool,
      &opt_bptt_adaptive_min, "don't auto-adapt BPTT minimum error threshold"),
  OPT_WITHOUT_ARG("-o|--override-params", opt_set_bool,
      &opt_override, "override meta-parameters in loaded net (where possible)"),
  OPT_WITH_ARG("-f|--filename=<file>", opt_set_charp, opt_show_charp, &opt_filename,
      "load/save net here"),
  OPT_WITH_ARG("--log-file=<file>", opt_set_charp, opt_show_charp, &opt_logfile,
      "log to this filename"),
  OPT_WITH_ARG("-t|--text-file=<file>", opt_set_charp, opt_show_charp, &opt_textfile,
      "learn from this text"),
  OPT_WITH_ARG("-t|--dump-collapsed-text=<file>", opt_set_charp, opt_show_charp,
      &opt_dump_collapsed_text, "dump internal text representation here"),
  OPT_WITH_ARG("-A|--alphabet=<chars>", opt_set_charp, opt_show_charp, &opt_alphabet,
      "Use only these characters"),
  OPT_WITH_ARG("-C|--collapse-chars=<chars>", opt_set_charp, opt_show_charp,
      &opt_collapse_chars, "Map these characters to first in alphabet"),
  OPT_WITHOUT_ARG("--temporal-pgm-dump", opt_set_bool,
      &opt_temporal_pgm_dump, "Dump ppm images showing inputs change over time"),
  OPT_WITHOUT_ARG("--periodic-pgm-dump", opt_set_bool,
      &opt_periodic_pgm_dump, "Dump ppm images of weights, every reporting interval"),
  OPT_WITHOUT_ARG("--learn-capitals", opt_set_bool,
      &opt_learn_capitals, "learn to predict capitalisation"),
  OPT_WITHOUT_ARG("--deterministic-confab", opt_set_bool,
      &opt_deterministic_confab, "Use best guess in confab, not random sampling"),
  OPT_WITHOUT_ARG("--no-save-net", opt_set_invbool,
      &opt_save_net, "Don't save learnt changes"),
  OPT_WITH_ARG("--dropout=<0-1>", opt_set_floatval, opt_show_floatval,
      &opt_dropout, "dropout this fraction of hidden nodes"),
  OPT_WITH_ARG("--multi-tap=<n>", opt_set_uintval, opt_show_uintval,
      &opt_multi_tap, "read at n evenly spaced points in parallel"),
  OPT_WITHOUT_ARG("--use-multi-tap-path", opt_set_bool,
      &opt_use_multi_tap_path, "use multi-tap code path on single-tap tasks"),
  OPT_WITH_ARG("--momentum-style=<n>", opt_set_intval, opt_show_intval,
      &opt_momentum_style, "0: weighted, 1: Nesterov, 2: simplified N., 3: classical"),
  OPT_WITH_ARG("--weight-scale-factor=<float>", opt_set_floatval, opt_show_floatval,
      &opt_weight_scale_factor, "scale newly initialised weights (try ~0.5)"),
  OPT_WITH_ARG("--report-interval=<n>", opt_set_uintval_bi, opt_show_uintval_bi,
      &opt_report_interval, "how often to validate and report"),
  OPT_WITH_ARG("--confab-only=<chars>", opt_set_uintval, opt_show_uintval,
      &opt_confab_only, "no training, only confabulate this many characters"),
  OPT_WITH_ARG("--diagonal-boost=<float>", opt_set_floatval, opt_show_floatval,
      &opt_diagonal_boost, "boost this portion of diagonal weights"),


  OPT_WITHOUT_ARG("-h|--help", opt_usage_and_exit,
      ": Rnn modelling of text at the character level",
      "Print this message."),
  OPT_ENDTABLE
};

static inline float
capped_log2f(float x){
  return (x < 1e-30f) ? -100.0f : log2f(x);
}

static u8*
new_char_lut(const char *alphabet, const u8 *collapse_chars){
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
      if (opt_learn_capitals){
        ctn[c - 32] = i | 0x80;
      }
      else {
        ctn[c - 32] = i;
      }
    }
  }
  return ctn;
}

static inline u8*
alloc_and_collapse_text(char *filename, const char *alphabet, const u8 *collapse_chars,
    long *len){
  int i, j;
  u8 *char_to_net = new_char_lut(alphabet, collapse_chars);
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
  Q_DEBUG(1, "original text was %d chars, collapsed is %d", i, j);
  err |= fclose(f);
  if (err){
    Q_DEBUG(2, "something went wrong with the file %p (%s). error %d",
    f, filename, err);
  }
  return text;
}

static inline void
dump_collapsed_text(u8 *text, int len, char *name)
{
  int i;
  FILE *f = fopen_or_abort(name, "w");
  for (i = 0; i < len; i++){
    u8 c = text[i];
    if (c & 0x80){
      fputc(toupper(opt_alphabet[c & 0x7f]), f);
    }
    else{
      fputc(opt_alphabet[c], f);
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
one_hot_opinion(RecurNN *net, const int hot){
  //XXX could just set the previous one to zero (i.e. remember it)
  memset(net->real_inputs, 0, net->input_size * sizeof(float));
  if (opt_learn_capitals && (hot & 0x80)){
    net->real_inputs[net->input_size - 1] = 1.0f;
  }
  net->real_inputs[hot & 0x7f] = 1.0f;
  return rnn_opinion(net, NULL, opt_dropout);
}

static inline float
net_error_bptt(RecurNN *net, float *restrict error, int c, int next, int *correct){
  ASSUME_ALIGNED(error);
  float *answer = one_hot_opinion(net, c);
  int winner;
  if (opt_learn_capitals){
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


static inline int
opinion_deterministic(RecurNN *net, int hot){
  float *answer = one_hot_opinion(net, hot);
  if (opt_learn_capitals){
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

static inline int
opinion_probabilistic(RecurNN *net, int hot){
  int i;
  float r;
  float *answer = one_hot_opinion(net, hot);
  int n_chars = net->output_size - (opt_learn_capitals ? 2 : 0);
  int cap = 0;
  float error[net->output_size];
  if (opt_learn_capitals){
    r = rand_double(&net->rng);
    softmax(error, answer + n_chars, 2);
    if (r > error[0])
      cap = 0x80;
  }
  softmax(error, answer, n_chars);
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

static float
validate(RecurNN *net, const u8 *text, int len){
  float error[net->output_size];
  float entropy = 0.0f;
  int i;
  int n_chars = net->output_size - (opt_learn_capitals ? 2 : 0);
  /*skip the first few because state depends too much on previous experience */
  int skip = MIN(len / 10, 5);
  for (i = 0; i < skip; i++){
    one_hot_opinion(net, text[i]);
  }
  for (; i < len - 1; i++){
    float *answer = one_hot_opinion(net, text[i]);
    softmax(error, answer, n_chars);
    float e = error[text[i + 1] & 0x7f];
    entropy += capped_log2f(e);
  }
  entropy /= -(len - skip - 1);
  return entropy;
}

static void
confabulate(RecurNN *net, char *text, int len,
    int deterministic){
  int i;
  static int n = 0;
  for (i = 0; i < len; i++){
    if (deterministic)
      n = opinion_deterministic(net, n);
    else
      n = opinion_probabilistic(net, n);
    int cap = n & 0x80;
    n &= 0x7f;
    int c = opt_alphabet[n];
    if (cap){
      c = toupper(c);
    }
    text[i] = c;
  }
}

static inline void
long_confab(RecurNN *net, int len, int rows){
  int i, j;
  char confab[len * rows + 1];
  confab[len * rows] = 0;
  confabulate(net, confab, len * rows, 0);
  for (i = 1; i < rows; i++){
    int target = i * len;
    int linebreak = target;
    int best = 100;
    for (j = MAX(target - 12, 1);
         j < MIN(target + 12, len * rows - 1); j++){
      if (confab[j] == ' '){
        int d = abs(j - target);
        if ( d < best){
          best = d;
          linebreak = j;
        }
      }
    }
    confab[linebreak] = '\n';
    if (best == 100){
      confab[linebreak - 1] = '}';
      confab[linebreak + 1] = '{';
    }
  }
  Q_DEBUG(1, "%s", confab);
}

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

static inline void
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

static inline float
calc_ventropy(Ventropy *v, int lap)
{
  if (v->len > 0){
    if (v->lap > 1 && lap){
      v->counter++;
      if (v->counter == v->lap){
        v->counter = 0;
      }
      v->history[v->counter] = validate(v->net, v->text + v->lapsize * v->counter,
          v->lapsize);
      float sum = 0.0f;
      float div = v->lap;
      for (int j = 0; j < v->lap; j++){
        div -= v->history[j] == 0;
        sum += v->history[j];
      }
      v->entropy = div ? sum / div : 0;
    }
    else {
      v->entropy = validate(v->net, v->text, v->len);
      v->history[0] = v->entropy;
    }
  }
  return v->entropy;
}

static inline void
finish(RecurNN *net, Ventropy *v){
  if (opt_filename && opt_save_net){
    rnn_save_net(net, opt_filename, 1);
  }
  BELOW_QUIET_LEVEL(2){
    float ventropy = calc_ventropy(v, 0);
    DEBUG("final entropy %.3f; learn rate %.2g; momentum %.2g",
        ventropy, net->bptt->learn_rate, net->bptt->momentum);
  }
  exit(0);
}

static inline void
report_on_progress(RecurNN *net, RecurNN *confab_net, float ventropy,
    int *correct, float *error, float *entropy, double elapsed, float scale){
  char confab[CONFAB_SIZE + 1];
  confab[CONFAB_SIZE] = 0;
  int k = net->generation >> 10;
  *entropy *= -scale;
  *error *= scale;
  float accuracy = *correct * scale;
  double per_sec = 1.0 / scale / elapsed;
  BELOW_QUIET_LEVEL(1){
    confabulate(confab_net, confab, CONFAB_SIZE, opt_deterministic_confab);
    Q_DEBUG(1, "%5dk e.%02d t%.2f v%.2f a.%02d %.0f/s |%s|", k, (int)(*error * 100 + 0.5),
        *entropy, ventropy,
        (int)(accuracy * 100 + 0.5), per_sec + 0.5, confab);
  }
  rnn_log_float(net, "error", *error);
  rnn_log_float(net, "t_entropy", *entropy);
  rnn_log_float(net, "v_entropy", ventropy);
  rnn_log_float(net, "momentum", net->bptt->momentum);
  rnn_log_float(net, "accuracy", accuracy);
  rnn_log_float(net, "learn-rate", net->bptt->learn_rate);
  rnn_log_float(net, "per_second", per_sec);
  *correct = 0;
  *error = 0.0f;
  *entropy = 0.0f;
}

static void
epoch(RecurNN **nets, int n_nets, RecurNN *confab_net, Ventropy *v,
    Schedule *schedule,
    const u8 *text, const int len,
    const int start){
  int i, j;
  float error = 0.0f;
  float entropy = 0.0f;
  int correct = 0;
  float e;
  int c;
  int spacing = (len - 1) / n_nets;
  RecurNN *net = nets[0];
  uint report_counter = net->generation % opt_report_interval;
  struct timespec timers[2];
  struct timespec *time_start = timers;
  struct timespec *time_end = timers + 1;
  clock_gettime(CLOCK_MONOTONIC, time_start);
  for(i = start; i < len - 1; i++){
    float momentum = rnn_calculate_momentum_soft_start(net->generation,
        opt_momentum, opt_momentum_soft_start);
    if (n_nets > 1 || opt_momentum_style != RNN_MOMENTUM_WEIGHTED ||
        opt_use_multi_tap_path){
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
      rnn_apply_learning(net, opt_momentum_style, momentum);
    }
    else {
      RecurNNBPTT *bptt = net->bptt;
      bptt->momentum = momentum;
      rnn_bptt_advance(net);
      e = net_error_bptt(net, bptt->o_error, text[i], text[i + 1], &c);
      rnn_bptt_calculate(net, opt_batch_size);
      correct += c;
      error += e;
      entropy += capped_log2f(1.0f - e);
    }

    if (opt_temporal_pgm_dump){
      temporal_ppm_add_row(input_ppm, net->input_layer);
      temporal_ppm_add_row(error_ppm, net->bptt->o_error);
    }
    report_counter++;
    if (report_counter >= opt_report_interval){
      report_counter = 0;
      clock_gettime(CLOCK_MONOTONIC, time_end);
      s64 secs = time_end->tv_sec - time_start->tv_sec;
      s64 nano = time_end->tv_nsec - time_start->tv_nsec;
      double elapsed = secs + 1e-9 * nano;
      struct timespec *tmp = time_end;
      time_end = time_start;
      time_start = tmp;
      float ventropy = calc_ventropy(v, 1);
      report_on_progress(net, confab_net, ventropy, &correct, &error, &entropy,
          elapsed, 1.0f / (opt_report_interval * n_nets));
      if (opt_save_net && opt_filename){
        rnn_save_net(net, opt_filename, 1);
      }
      if (opt_periodic_pgm_dump){
        rnn_multi_pgm_dump(net, "ihw how ihd hod ihm hom");
      }
      schedule->eval(schedule, net, ventropy);
    }
    if (opt_stop && (int)net->generation >= opt_stop){
      finish(net, v);
    }
  }
  BELOW_QUIET_LEVEL(1){
    long_confab(confab_net, CONFAB_SIZE, 6);
  }
}


static char*
construct_net_filename(void){
  char s[260];
  int alpha_size = strlen(opt_alphabet);
  int input_size = alpha_size + (opt_learn_capitals ? 1 : 0);
  int output_size = alpha_size + (opt_learn_capitals ? 2 : 0);
  uint sig = 0;
  snprintf(s, sizeof(s), "%s--%s", opt_alphabet, opt_collapse_chars);
  uint len = strlen(s);
  for (uint i = 0; i < len; i++){
    sig ^= ROTATE(sig - s[i], 13) + s[i];
  }
  snprintf(s, sizeof(s), "text-s%0x-i%d-h%d-o%d-b%d-c%d.net",
      sig, input_size, opt_hidden_size, output_size,
      opt_bias, opt_learn_capitals);
  DEBUG("filename: %s", s);
  return strdup(s);
}

static RecurNN *
load_or_create_net(void){
  RecurNN *net = NULL;
  if (opt_filename == NULL){
    opt_filename = construct_net_filename();
  }
  if (opt_reload){
    net = rnn_load_net(opt_filename);
    if (net){
      rnn_set_log_file(net, opt_logfile, 1);
    }
  }
  if (net == NULL){
    int input_size = strlen(opt_alphabet);
    int output_size = input_size;
    if (opt_learn_capitals){
      input_size++;
      output_size += 2;
    }
    u32 flags = opt_bias ? RNN_NET_FLAG_STANDARD : RNN_NET_FLAG_NO_BIAS;
    if (opt_bptt_adaptive_min){/*on by default*/
      flags |= RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR;
    }
    net = rnn_new(input_size, opt_hidden_size,
        output_size, flags, opt_rng_seed,
        opt_logfile, opt_bptt_depth, opt_learn_rate,
        opt_momentum);
    if (opt_dense_weights){
      rnn_randomise_weights(net, RNN_INITIAL_WEIGHT_VARIANCE_FACTOR / net->h_size,
          opt_dense_weights, opt_perforate_weights);
    }
    else {
      rnn_randomise_weights_auto(net);
      //rnn_randomise_weights_fan_in(net, 2.0f, 0.3f, 0.1f, 1.0);
    }
    if (opt_diagonal_boost){
      rnn_emphasise_diagonal(net, 0.5, opt_diagonal_boost);
    }
    net->bptt->momentum_weight = opt_momentum_weight;
    if (opt_weight_scale_factor > 0){
      rnn_scale_initial_weights(net, opt_weight_scale_factor);
    }
    if (opt_periodic_pgm_dump){
      rnn_multi_pgm_dump(net, "ihw how");
    }
  }
  else if (opt_override){
    RecurNNBPTT *bptt = net->bptt;
    bptt->learn_rate = opt_learn_rate;
    bptt->momentum = opt_momentum;
    bptt->momentum_weight = opt_momentum_weight;
  }

  return net;
}


int
main(int argc, char *argv[]){
  //feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  opt_register_table(options, NULL);
  if (!opt_parse(&argc, argv, opt_log_stderr)){
    exit(1);
  }
  if (argc > 1){
    Q_DEBUG(1, "unused arguments:");
    for (int i = 1; i < argc; i++){
      Q_DEBUG(1, "   '%s'", argv[i]);
    }
    opt_usage(argv[0], NULL);
  }
  opt_multi_tap = MAX(opt_multi_tap, 1);
  RecurNN *net = load_or_create_net();
  if (opt_confab_only){
    long_confab(net, opt_confab_only, 1);
    exit(0);
  }

  RecurNN **nets;

  if (opt_multi_tap > 1){
    nets = rnn_new_training_set(net, opt_multi_tap);
  }
  else{
    nets = &net;
  }

  RecurNN *confab_net = rnn_clone(net,
      net->flags & ~(RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS),
      RECUR_RNG_SUBSEED,
      NULL);
  RecurNN *validate_net = rnn_clone(net,
      net->flags & ~(RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS),
      RECUR_RNG_SUBSEED,
      NULL);

  Schedule schedule;
  init_schedule(&schedule, opt_learn_rate_inertia, 0, opt_learn_rate_min,
      opt_learn_rate_scale);
  long len;
  u8* validate_text;
  u8* text = alloc_and_collapse_text(opt_textfile,
      opt_alphabet, (u8 *)opt_collapse_chars, &len);
  if (opt_dump_collapsed_text){
    dump_collapsed_text(text, len, opt_dump_collapsed_text);
  }

  if (opt_validate_chars > 2){
    len -= opt_validate_chars;
    validate_text = text + len;
  }
  else {
    if (opt_validate_chars){
      DEBUG("--validate-chars needs to be bigger");
      opt_validate_chars = 0;
    }
    validate_text = NULL;
  }
  int start_char;
  if (opt_start_char >= 0 && opt_start_char < len - 1){
    start_char = opt_start_char;
  }
  else {
    start_char = net->generation % len;
  }

  if (opt_temporal_pgm_dump){
    input_ppm = temporal_ppm_alloc(net->i_size, 300, "input_layer", 0, PGM_DUMP_COLOUR,
        NULL);
    error_ppm = temporal_ppm_alloc(net->o_size, 300, "output_error", 0, PGM_DUMP_COLOUR,
        NULL);
  }
  Ventropy v;

  init_ventropy(&v, validate_net, validate_text,
      opt_validate_chars, opt_validation_overlap);

  if (opt_stop < 0){
    opt_stop = net->generation - opt_stop;
  }

  BELOW_QUIET_LEVEL(2){
    START_TIMER(run);
    for (int i = 0;;i++){
      DEBUG("Starting epoch %d. learn rate %g.", i, net->bptt->learn_rate);
      START_TIMER(epoch);
      epoch(nets, opt_multi_tap, confab_net, &v, &schedule,
          text, len, start_char);
      DEBUG_TIMER(epoch);
      DEBUG_TIMER(run);
      start_char = 0;
    }
  }
  else {/* quiet level 2+ */
    for (;;){
      epoch(nets, opt_multi_tap, confab_net, &v, &schedule,
          text, len, start_char);
      start_char = 0;
    }
  }

  free(text);
  if (opt_multi_tap < 2){
    rnn_delete_net(net);
  }
  else {
    rnn_delete_training_set(nets, opt_multi_tap, 0);
  }
  rnn_delete_net(confab_net);
  rnn_delete_net(validate_net);

  temporal_ppm_free(input_ppm);
  temporal_ppm_free(error_ppm);
}
