#define DISABLE_PGM_DUMP 0
#define EXCESSIVE_PGM_DUMP 0
#define PERIODIC_PGM_DUMP 0
#define TEMPORAL_PGM_DUMP 0
#define CONFAB_HIDDEN_IMG 0
#define DETERMINISTIC_CONFAB 0
#define PERIODIC_SAVE_NET 1
#define TRY_RELOAD 0

#define NET_LOG_FILE "bptt.log"
#include "test-common.h"
#include "badmaths.h"
#include "ccan/opt/opt.h"
#include <errno.h>
#include <stdio.h>
#include <fenv.h>
#include <ctype.h>

#define BPTT_DEPTH 20
#define CONFAB_SIZE 80
#define LEARN_RATE 0.001
#define LEARN_RATE_DECAY 0.96
#define MIN_LEARN_RATE 1e-6

#define MOMENTUM 0.95
#define MOMENTUM_WEIGHT 0.5
#define BIAS 1
#define K_STOP 0
#define BPTT_BATCH_SIZE 1

#define Q_DEBUG(quiet, ...) do {                                \
                                if (quiet >= opt_quiet)         \
                                  STDERR_DEBUG(__VA_ARGS__); \
                                } while(0)

u8 CHAR_TO_NET[257];
const u8 NET_TO_CHAR[] = "abcdefghijklmnopqrstuvwxyz,'- .\"#;:!?";
const u8 HASH_CHARS[] = "1234567890&@";

#define HIDDEN_SIZE 199
#define INPUT_SIZE (sizeof(NET_TO_CHAR) - 1)

#define NET_FILENAME ("test_backprop-" QUOTE(HIDDEN_SIZE) "-" QUOTE(BIAS) ".net")

static uint opt_hidden_size = HIDDEN_SIZE;
static uint opt_bptt_depth = BPTT_DEPTH;
static float opt_learn_rate = LEARN_RATE;
static float opt_momentum = MOMENTUM;
static int opt_quiet = 0;
static char * opt_filename = TRY_RELOAD ? NET_FILENAME : NULL;

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
  OPT_WITH_ARG("-h|--hidden-size=<n>", opt_set_uintval, opt_show_uintval,
      &opt_hidden_size, "number of hidden nodes (" QUOTE(HIDDEN_SIZE) ")"),
  OPT_WITH_ARG("-d|--depth=<n>", opt_set_uintval, opt_show_uintval,
      &opt_bptt_depth, "max depth of BPTT recursion (" QUOTE(BPTT_DEPTH) ")"),
  OPT_WITH_ARG("-l|--learn-rate=<float>", opt_set_floatval, opt_show_floatval,
      &opt_learn_rate, "learning rate (" QUOTE(LEARN_RATE) ")"),
  OPT_WITH_ARG("-m|--momentum=<float>", opt_set_floatval, opt_show_floatval,
      &opt_momentum, "momentum (" QUOTE(MOMENTUM) ")"),
  OPT_WITHOUT_ARG("-q|--quiet", opt_inc_intval,
      &opt_quiet, "print less (twice for even less)"),
  OPT_WITH_ARG("-f|--filename=<file>", opt_set_charp, opt_show_charp, &opt_filename,
      TRY_RELOAD ? "filename (" QUOTE(NET_FILENAME) ")" : "filename (None)"),

  OPT_ENDTABLE
};

static int
create_char_lut(u8 *ctn, const u8 *ntc, const u8 *hash_chars){
  int i;
  int len = strlen((char *)ntc);
  int hash = strchr((char *)ntc, '#') - (char *)ntc;
  int space = strchr((char *)ntc, ' ') - (char *)ntc;
  memset(ctn, space, 257);
  for (i = 0; hash_chars[i]; i++){
    ctn[hash_chars[i]] = hash;
  }
  for (i = 0; i < len; i++){
    u8 c = ntc[i];
    ctn[c] = i;
    if (islower(c))
      ctn[c - 32] = i;
  }
  return len;
}

static inline u8*
alloc_and_collapse_text(char *filename, long *len){
  int i, j;
  FILE *f = fopen(filename, "r");
  int err = fseek(f, 0, SEEK_END);
  *len = ftell(f);
  err |= fseek(f, 0, SEEK_SET);
  u8 *text = malloc(*len + 1);
  u8 prev = 0;
  u8 c;
  int chr = 0;
  int space = CHAR_TO_NET[' '];
  j = 0;
  for(i = 0; i < *len && chr != EOF; i++){
    chr = getc(f);
    c = CHAR_TO_NET[chr];
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

static void UNUSED
dump_collapsed_text(u8 *text, int len, char *name)
{
  int i;
  FILE *f = fopen(name, "w");
  for (i = 0; i < len; i++){
    fputc(NET_TO_CHAR[text[i]], f);
  }
  fclose(f);
}


static inline float*
one_hot_opinion(RecurNN *net, const int hot){
  memset(net->real_inputs, 0, INPUT_SIZE * sizeof(float));
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

  if (EXCESSIVE_PGM_DUMP){
    dump_colour_weights_autoname(net->ho_weights, net->o_size, net->h_size,
        "ho", net->generation);
  }
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
    int hidden_ppm, int deterministic){
  int i;
  float *im;
  if (hidden_ppm){
    im = malloc_aligned_or_die(net->h_size * len * sizeof(float));
  }
  static int n = 0;
  for (i = 0; i < len; i++){
    if (deterministic)
      n = opinion_deterministic(net, n);
    else
      n = opinion_probabilistic(net, n);
    int c = NET_TO_CHAR[n];
    text[i] = c;
    if (hidden_ppm)
      memcpy(&im[net->h_size * i], net->hidden_layer, net->h_size * sizeof(float));
  }
  if (hidden_ppm){
    dump_colour_weights_autoname(im, net->h_size, len,
      "confab-hiddens", net->generation);
    free(im);
  }
}

static inline void
long_confab(RecurNN *net, int len, int rows, int hidden_ppm){
  int i, j;
  char confab[len * rows + 1];
  confab[len * rows] = 0;
  confabulate(net, confab, len * rows, hidden_ppm, 0);
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
      confabulate(confab_net, confab, CONFAB_SIZE, CONFAB_HIDDEN_IMG,
          DETERMINISTIC_CONFAB);
      Q_DEBUG(0, "%4dk .%02d %.2f .%02d |%s|", k, (int)(error / 10.24f + 0.5), entropy,
          (int)(correct / 10.24f + 0.5), confab);
      bptt_log_float(net, "error", error / 1024.0f);
      bptt_log_float(net, "entropy", entropy);
      bptt_log_float(net, "accuracy", correct / 1024.0f);
      correct = 0;
      error = 0.0f;
      entropy = 0.0f;
      if (PERIODIC_SAVE_NET){
        rnn_save_net(net, NET_FILENAME);
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
  long_confab(confab_net, CONFAB_SIZE, 6, CONFAB_HIDDEN_IMG);
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
  }
  RecurNN *net = NULL;
  if (opt_filename){
    net = rnn_load_net(opt_filename);
    if (net){
      rnn_set_log_file(net, NET_LOG_FILE, 0);
    }
  }
  if (net == NULL){
    u32 flags = BIAS ? RNN_NET_FLAG_STANDARD : RNN_NET_FLAG_NO_BIAS;
    net = rnn_new(INPUT_SIZE, opt_hidden_size,
        INPUT_SIZE, flags, 1,
        NET_LOG_FILE, opt_bptt_depth, opt_learn_rate, opt_momentum, MOMENTUM_WEIGHT,
        BPTT_BATCH_SIZE);
  }
  RecurNN *confab_net = rnn_clone(net,
      net->flags & ~(RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS),
      RECUR_RNG_SUBSEED,
      NULL);

  create_char_lut(CHAR_TO_NET, NET_TO_CHAR, HASH_CHARS);
  long len;
  u8* text = alloc_and_collapse_text(DICKENS_SHUFFLED_TEXT, &len);
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
