#define PERIODIC_PGM_DUMP 0
#define TEMPORAL_PGM_DUMP 0
#define DETERMINISTIC_CONFAB 0
#define PERIODIC_SAVE_NET 1
#define DEFAULT_RELOAD 0

#define NET_LOG_FILE "bptt.log"
#include "test-common.h"
#include "badmaths.h"
#include "ccan/opt/opt.h"
#include <errno.h>
#include <stdio.h>
#include <fenv.h>
#include <ctype.h>

#define DEFAULT_BPTT_DEPTH 30
#define CONFAB_SIZE 80
#define DEFAULT_LEARN_RATE 0.001
#define LEARN_RATE_DECAY 0.96
#define MIN_LEARN_RATE 1e-6

#define DEFAULT_MOMENTUM 0.95
#define DEFAULT_MOMENTUM_WEIGHT 0.5
#define DEFAULT_BIAS 1
#define DEFAULT_RNG_SEED 1
#define K_STOP 0
#define BPTT_BATCH_SIZE 1

#define Q_DEBUG(quiet, ...) do {                                \
                                if (quiet >= opt_quiet)         \
                                  STDERR_DEBUG(__VA_ARGS__); \
                                } while(0)

#define NET_TO_CHAR "#abcdefghijklmnopqrstuvwxyz,'- .\";:!?"
#define HASH_CHARS "1234567890&@"

#define HIDDEN_SIZE 199

static uint opt_hidden_size = HIDDEN_SIZE;
static uint opt_bptt_depth = DEFAULT_BPTT_DEPTH;
static float opt_learn_rate = DEFAULT_LEARN_RATE;
static float opt_momentum = DEFAULT_MOMENTUM;
static int opt_quiet = 0;
static char * opt_filename = NULL;
static char * opt_logfile = NET_LOG_FILE;
static char * opt_alphabet = NET_TO_CHAR;
static char * opt_collapse_chars = HASH_CHARS;
static char * opt_textfile = DICKENS_SHUFFLED_TEXT;
static bool opt_bias = DEFAULT_BIAS;
static bool opt_reload = DEFAULT_RELOAD;
static float opt_momentum_weight = DEFAULT_MOMENTUM_WEIGHT;
static u64 opt_rng_seed = DEFAULT_RNG_SEED;


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
  OPT_WITH_ARG("-r|--rng-seed=<n>", opt_set_ulongval_bi, opt_show_ulongval_bi,
      &opt_rng_seed, "RNG seed (-1 for auto)"),
  OPT_WITH_ARG("-l|--learn-rate=<float>", opt_set_floatval, opt_show_floatval,
      &opt_learn_rate, "learning rate"),
  OPT_WITH_ARG("-m|--momentum=<float>", opt_set_floatval, opt_show_floatval,
      &opt_momentum, "momentum"),
  OPT_WITHOUT_ARG("-q|--quiet", opt_inc_intval,
      &opt_quiet, "print less (twice for even less)"),
  OPT_WITHOUT_ARG("--bias", opt_set_bool,
      &opt_bias, "use bias (default)"),
  OPT_WITHOUT_ARG("--no-bias", opt_set_invbool,
      &opt_bias, "Don't use bias"),
  OPT_WITHOUT_ARG("--reload", opt_set_bool,
      &opt_reload, "try to reload the net"),
  OPT_WITHOUT_ARG("-N|--no-reload", opt_set_invbool,
      &opt_reload, "Don't try to reload"),
  OPT_WITH_ARG("-f|--filename=<file>", opt_set_charp, opt_show_charp, &opt_filename,
      "load/save net here"),
  OPT_WITH_ARG("--log-file=<file>", opt_set_charp, opt_show_charp, &opt_logfile,
      "log to this filename"),
  OPT_WITH_ARG("-t|--text-file=<file>", opt_set_charp, opt_show_charp, &opt_textfile,
      "learn from this text"),
  OPT_WITH_ARG("-A|--alphabet=<chars>", opt_set_charp, opt_show_charp, &opt_alphabet,
      "Use only these characters"),
  OPT_WITH_ARG("-C|--collapse-chars=<chars>", opt_set_charp, opt_show_charp,
      &opt_collapse_chars, "Map these characters to first in alphabet"),


  OPT_WITHOUT_ARG("-h|--help", opt_usage_and_exit,
      ": Rnn modelling of text at the character level",
      "Print this message."),
  OPT_ENDTABLE
};

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
    if (islower(c))
      ctn[c - 32] = i;
  }
  return ctn;
}

static inline u8*
alloc_and_collapse_text(char *filename, const char *alphabet, const u8 *collapse_chars,
    long *len){
  int i, j;
  u8 *char_to_net = new_char_lut(alphabet, collapse_chars);
  FILE *f = fopen(filename, "r");
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
  FILE *f = fopen(name, "w");
  for (i = 0; i < len; i++){
    fputc(opt_alphabet[text[i]], f);
  }
  fclose(f);
}


static inline float*
one_hot_opinion(RecurNN *net, const int hot){
  memset(net->real_inputs, 0, net->input_size * sizeof(float));
  net->real_inputs[hot] = 1.0f;
  float *answer = rnn_opinion(net, NULL);
  //net->real_inputs[hot] = 0.0;
  return answer;
}

static inline float
net_error_bptt(RecurNN *net, float *restrict error, int c, int next, int *correct){
  ASSUME_ALIGNED(error);
  float *answer = one_hot_opinion(net, c);
  int winner = softmax_best_guess(error, answer, net->output_size);
  *correct = (winner == next);
  error[next] += 1.0f;
  return error[next];
}

static void
sgd_one(RecurNN *net, const int current, const int next, float *error, int *correct){
  RecurNNBPTT *bptt = net->bptt;
  float sum;
  bptt_advance(net);
  sum = net_error_bptt(net, bptt->o_error, current, next, correct);

  bptt_calculate(net);

  *error = sum;
}


static inline int
opinion_deterministic(RecurNN *net, int hot){
  float *answer = one_hot_opinion(net, hot);
  return search_for_max(answer, net->output_size);
}

static inline int
opinion_probabilistic(RecurNN *net, int hot){
  int i;
  float r;
  float *answer = one_hot_opinion(net, hot);
  float error[net->output_size];
  softmax(error, answer, net->output_size);

  /*outer loop in case error doesn't quite add to 1 */
  for(;;){
    r = rand_double(&net->rng);
    float accum = 0.0;
    for (i = 0; i < net->output_size; i++){
      accum += error[i];
      if (r < accum)
        return i;
    }
  }
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
    int c = opt_alphabet[n];
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
  Q_DEBUG(0, "%s", confab);
}

static TemporalPPM *input_ppm;

void
epoch(RecurNN *net, RecurNN *confab_net, const u8 *text, const int len){
  int i;
  char confab[CONFAB_SIZE + 1];
  confab[CONFAB_SIZE] = 0;
  float error = 0.0f;
  float entropy = 0.0f;
  int correct = 0;
  for(i = 0; i < len - 1; i++){
    float e;
    int c;
    sgd_one(net, text[i], text[i + 1], &e, &c);
    correct += c;

    error += e;
    if (e < 1.0f)
      entropy += log2f(1.0f - e);
    else
      entropy -= 50;

    if (TEMPORAL_PGM_DUMP){
      temporal_ppm_add_row(input_ppm, net->input_layer);
    }

    if ((net->generation & 1023) == 0){
      int k = net->generation >> 10;
      entropy /= -1024.0f;
      confabulate(confab_net, confab, CONFAB_SIZE, DETERMINISTIC_CONFAB);
      Q_DEBUG(0, "%4dk .%02d %.2f .%02d |%s|", k, (int)(error / 10.24f + 0.5), entropy,
          (int)(correct / 10.24f + 0.5), confab);
      bptt_log_float(net, "error", error / 1024.0f);
      bptt_log_float(net, "entropy", entropy);
      bptt_log_float(net, "accuracy", correct / 1024.0f);
      correct = 0;
      error = 0.0f;
      entropy = 0.0f;
      if (PERIODIC_SAVE_NET && opt_filename){
        rnn_save_net(net, opt_filename);
      }
      if (PERIODIC_PGM_DUMP){
        rnn_multi_pgm_dump(net, "ihw how");
      }
      if (K_STOP && k > K_STOP)
        exit(0);
      if ((k & 1023) == 1023){
        net->bptt->learn_rate = MAX(MIN_LEARN_RATE,
            net->bptt->learn_rate * LEARN_RATE_DECAY);
      }
    }
  }
  long_confab(confab_net, CONFAB_SIZE, 6);
}

static char*
construct_net_filename(void){
  char s[260];
  int alpha_size = strlen(opt_alphabet);
  uint sig = 0;
  snprintf(s, sizeof(s), "%s--%s", opt_alphabet, opt_collapse_chars);
  for (uint i = 0; i < strlen(s); i++){
    sig ^= ROTATE(sig - s[i], 13) + s[i];
  }
  snprintf(s, sizeof(s), "text-s%0x-i%d-h%d-o%d-b%d.net",
      sig, alpha_size, opt_hidden_size, alpha_size,
      opt_bias);
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
      rnn_set_log_file(net, opt_logfile, 0);
    }
  }
  if (net == NULL){
    int input_size = strlen(opt_alphabet);
    u32 flags = opt_bias ? RNN_NET_FLAG_STANDARD : RNN_NET_FLAG_NO_BIAS;
    net = rnn_new(input_size, opt_hidden_size,
        input_size, flags, 1,
        opt_logfile, opt_bptt_depth, opt_learn_rate, opt_momentum, opt_momentum_weight,
        BPTT_BATCH_SIZE);
  }
  return net;
}


int
main(int argc, char *argv[]){
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  //feenableexcept(FE_ALL_EXCEPT & ~ FE_INEXACT);
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

  RecurNN *net = load_or_create_net();
  RecurNN *confab_net = rnn_clone(net,
      net->flags & ~(RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS),
      RECUR_RNG_SUBSEED,
      NULL);

  long len;
  u8* text = alloc_and_collapse_text(DICKENS_SHUFFLED_TEXT,
      opt_alphabet, (u8 *)opt_collapse_chars, &len);
  if (TEMPORAL_PGM_DUMP){
    input_ppm = temporal_ppm_alloc(net->i_size, 500, "input_layer", 0, PGM_DUMP_COLOUR);
  }
  START_TIMER(run);
  for (int i = 0; i < 200; i++){
    Q_DEBUG(1, "Starting epoch %d. learn rate %f.", i, net->bptt->learn_rate);
    START_TIMER(epoch);
    epoch(net, confab_net, text, len);
    DEBUG_TIMER(epoch);
  }
  free(text);
  rnn_delete_net(net);
  temporal_ppm_free(input_ppm);
}
