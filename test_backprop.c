/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> GPL2+

This uses the RNN to predict the next character in a text sequence.

Unlike most of the Recur repository, this file is licensed under the GNU
General Public License, version 2 or greater. That is because it is linked to
ccan/opt which is also GPL2+ (originally GPL3+, but relicensed, see
http://git.ozlabs.org/?p=ccan;a=commit;h=79715b8).

Because of ccan/opt, --help will tell you something.
*/

#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include <math.h>
#include "path.h"
#include "badmaths.h"
#include "ccan/opt/opt.h"
#include <errno.h>
#include <stdio.h>
#include <fenv.h>
#include <ctype.h>
#include "charmodel.h"

#define PGM_DUMP_STRING "ihw how"

#define DICKENS_SHUFFLED_TEXT TEST_DATA_DIR "/dickens-shuffled.txt"
#define DICKENS_TEXT TEST_DATA_DIR "/dickens.txt"
#define EREWHON_TEXT TEST_DATA_DIR "/erewhon.txt"
#define EREWHON_LONG_TEXT TEST_DATA_DIR "/erewhon-erewhon"\
  "-revisited-sans-gutenberg.txt"

/*Default text and characters to use.

  To see what characters are in a text, use
    `./scripts/find-character-set $filename`.

  The characters in DEFAULT_COLLAPSE_CHARS get collapsed into the first
  character of DEFAULT_CHARSET. Typically it is used to avoid predicting
  digits in cases where digits are rare and largely random (e.g. the phone
  number and zip code of project gutenberg).
*/

#define DEFAULT_TEXT EREWHON_TEXT
#define DEFAULT_CHARSET "8 etaonihsrdlucmwfygpb,v.k-;x\"qj'?:z)(_!*&"
#define DEFAULT_COLLAPSE_CHARS "10872}{659/34][@"

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
#define DEFAULT_RNG_SEED 1
#define DEFAULT_STOP 0
#define DEFAULT_BATCH_SIZE 1
#define DEFAULT_VALIDATE_CHARS 0
#define DEFAULT_VALIDATION_OVERLAP 1
#define DEFAULT_OVERRIDE 0
#define DEFAULT_CONFAB_BIAS 0
#define DEFAULT_SAVE_NET 1
#define DEFAULT_LOG_FILE "bptt.log"
#define DEFAULT_BASENAME "text"
#define DEFAULT_START_CHAR -1
#define DEFAULT_HIDDEN_SIZE 199
#define DEFAULT_INIT_METHOD RNN_INIT_FLAT
#define DEFAULT_INIT_SUBMETHOD RNN_INIT_FLAT
#define DEFAULT_FLAT_INIT_DISTRIBUTION RNN_INIT_DIST_SEMICIRCLE
/*Init parameters are negative for automatic */
#define DEFAULT_INIT_VARIANCE -1.0f
#define DEFAULT_INIT_INPUT_PROBABILITY -1.0f
#define DEFAULT_INIT_INPUT_MAGNITUDE -1.0f
#define DEFAULT_INIT_HIDDEN_GAIN -1.0f
#define DEFAULT_INIT_HIDDEN_RUN_LENGTH -1.0f
#define DEFAULT_INIT_HIDDEN_RUN_DEVIATION -1.0f
#define DEFAULT_PERFORATE_WEIGHTS 0.0f

#define DEFAULT_LEARN_CAPITALS 0
#define DEFAULT_DUMP_COLLAPSED_TEXT NULL
#define DEFAULT_MULTI_TAP 0
#define DEFAULT_USE_MULTI_TAP_PATH 0
#define DEFAULT_MOMENTUM_STYLE RNN_MOMENTUM_WEIGHTED
#define DEFAULT_INIT_WEIGHT_SCALE 0
#define DEFAULT_REPORT_INTERVAL 1024
#define DEFAULT_CONFAB_ONLY 0
#define DEFAULT_BOTTOM_LAYER 0
#define DEFAULT_TOP_LEARN_RATE_SCALE 1.0f
#define DEFAULT_BOTTOM_LEARN_RATE_SCALE 1.0f
#define DEFAULT_PERIODIC_WEIGHT_NOISE 0

#define BELOW_QUIET_LEVEL(quiet) if (opt_quiet < quiet)

#define Q_DEBUG(quiet, ...) do {                               \
                                if (opt_quiet < quiet)         \
                                  STDERR_DEBUG(__VA_ARGS__); \
                                } while(0)


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
static char * opt_basename = DEFAULT_BASENAME;
static char * opt_alphabet = DEFAULT_CHARSET;
static char * opt_collapse_chars = DEFAULT_COLLAPSE_CHARS;
static char * opt_textfile = DEFAULT_TEXT;
static char * opt_dump_collapsed_text = DEFAULT_DUMP_COLLAPSED_TEXT;
static bool opt_reload = DEFAULT_RELOAD;
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
static int opt_init_method = DEFAULT_INIT_METHOD;
static int opt_init_submethod = DEFAULT_INIT_SUBMETHOD;
static int opt_flat_init_distribution = DEFAULT_FLAT_INIT_DISTRIBUTION;
static float opt_init_variance = DEFAULT_INIT_VARIANCE;
static float opt_init_input_probability = DEFAULT_INIT_INPUT_PROBABILITY;
static float opt_init_input_magnitude = DEFAULT_INIT_INPUT_MAGNITUDE;
static float opt_init_hidden_gain = DEFAULT_INIT_HIDDEN_GAIN;
static float opt_init_hidden_run_length = DEFAULT_INIT_HIDDEN_RUN_LENGTH;
static float opt_init_hidden_run_deviation = DEFAULT_INIT_HIDDEN_RUN_DEVIATION;
static float opt_init_weight_scale = DEFAULT_INIT_WEIGHT_SCALE;
static float opt_perforate_weights = DEFAULT_PERFORATE_WEIGHTS;
static bool opt_temporal_pgm_dump = DEFAULT_TEMPORAL_PGM_DUMP;
static bool opt_periodic_pgm_dump = DEFAULT_PERIODIC_PGM_DUMP;
static float opt_confab_bias = DEFAULT_CONFAB_BIAS;
static bool opt_save_net = DEFAULT_SAVE_NET;
static bool opt_learn_capitals = DEFAULT_LEARN_CAPITALS;
static uint opt_multi_tap = DEFAULT_MULTI_TAP;
static bool opt_use_multi_tap_path = DEFAULT_USE_MULTI_TAP_PATH;
static int opt_momentum_style = DEFAULT_MOMENTUM_STYLE;
static uint opt_report_interval = DEFAULT_REPORT_INTERVAL;
static uint opt_confab_only = DEFAULT_CONFAB_ONLY;
static uint opt_bottom_layer = DEFAULT_BOTTOM_LAYER;
static float opt_top_learn_rate_scale = DEFAULT_TOP_LEARN_RATE_SCALE;
static float opt_bottom_learn_rate_scale = DEFAULT_BOTTOM_LEARN_RATE_SCALE;
static float opt_periodic_weight_noise = DEFAULT_PERIODIC_WEIGHT_NOISE;

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

/*restrict to 0-1 range (mostly for probabilities)*/
static char *
opt_set_floatval01(const char *arg, float *f){
  char *msg = opt_set_floatval(arg, f);
  if (msg == NULL && (*f < 0.0f || *f > 1.0f)){
    char *s;
    if (asprintf(&s, "We want a number between 0 and 1, not '%s'", arg) > 0){
      return s;
    }
  }
  return msg;
}

static
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
  OPT_WITH_ARG("--init-method=<n>", opt_set_intval, opt_show_intval,
      &opt_init_method, "1: uniform-ish, 2: fan-in, 3: runs or loops"),
  OPT_WITH_ARG("--init-submethod=<n>", opt_set_intval, opt_show_intval,
      &opt_init_submethod, "initialisation for non-recurrent parts (1 or 2)"),
  OPT_WITH_ARG("--flat-init-distribution=<n>", opt_set_intval, opt_show_intval,
      &opt_flat_init_distribution, "1: uniform, 2: gaussian, 3: log-normal, 4: semicircle"
  ),
  OPT_WITH_ARG("--init-variance=<float>", opt_set_floatval, opt_show_floatval,
      &opt_init_variance, "variance of initial weights"),
  OPT_WITH_ARG("--init-input-probability=<0-1>", opt_set_floatval01, opt_show_floatval,
      &opt_init_input_probability, "chance of input weights"),
  OPT_WITH_ARG("--init-input-magnitude=<float>", opt_set_floatval, opt_show_floatval,
      &opt_init_input_magnitude, "stddev of input weight strength"),
  OPT_WITH_ARG("--init-hidden-gain=<float>", opt_set_floatval, opt_show_floatval,
      &opt_init_hidden_gain, "average strength of hidden weights (in runs)"),
  OPT_WITH_ARG("--init-hidden-run-length=<n>", opt_set_floatval, opt_show_floatval,
      &opt_init_hidden_run_length, "average length of hidden weight runs"),
  OPT_WITH_ARG("--init-hidden-run-deviation=<float>", opt_set_floatval, opt_show_floatval,
      &opt_init_hidden_run_deviation, "deviation of hidden weight run length"),


  OPT_WITH_ARG("--perforate-weights=<0-1>", opt_set_floatval01, opt_show_floatval,
      &opt_perforate_weights, "Zero this portion of weights"),
  OPT_WITH_ARG("-V|--validate-chars=<n>", opt_set_intval_bi, opt_show_intval_bi,
      &opt_validate_chars, "Retain this many characters for validation"),
  OPT_WITH_ARG("--validation-overlap=<n>", opt_set_intval, opt_show_intval,
      &opt_validation_overlap, "> 1 to use lapped validation (quicker)"),
  OPT_WITH_ARG("--start-char=<n>", opt_set_intval, opt_show_intval,
      &opt_start_char, "character to start epoch on (-1 for auto)"),
  OPT_WITH_ARG("-l|--learn-rate=<0-1>", opt_set_floatval01, opt_show_floatval,
      &opt_learn_rate, "initial learning rate"),
  OPT_WITH_ARG("--learn-rate-min=<0-1>", opt_set_floatval01, opt_show_floatval,
      &opt_learn_rate_min, "minimum learning rate (>learn-rate is off)"),
  OPT_WITH_ARG("--learn-rate-inertia=<int>", opt_set_intval, opt_show_intval,
      &opt_learn_rate_inertia, "tardiness of learn-rate reduction (try 30-90)"),
  OPT_WITH_ARG("--learn-rate-scale=<int>", opt_set_floatval, opt_show_floatval,
      &opt_learn_rate_scale, "size of learn rate reductions"),
  OPT_WITH_ARG("-m|--momentum=<0-1>", opt_set_floatval01, opt_show_floatval,
      &opt_momentum, "momentum"),
  OPT_WITH_ARG("--momentum-weight=<0-1>", opt_set_floatval01, opt_show_floatval,
      &opt_momentum_weight, "momentum weight"),
  OPT_WITH_ARG("--momentum-soft-start=<float>", opt_set_floatval, opt_show_floatval,
      &opt_momentum_soft_start, "softness of momentum onset (0 for constant)"),
  OPT_WITHOUT_ARG("-q|--quiet", opt_inc_intval,
      &opt_quiet, "print less (twice for even less)"),
  OPT_WITHOUT_ARG("-v|--verbose", opt_dec_intval,
      &opt_quiet, "print more, if possible"),
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
  OPT_WITH_ARG("-n|--basename=<tag>", opt_set_charp, opt_show_charp, &opt_basename,
      "construct log, image, net filenames from this root"),
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
  OPT_WITH_ARG("--confab-bias", opt_set_floatval, opt_show_floatval,
      &opt_confab_bias, "bias toward probable characters in confab (100 == deterministic)"),
  OPT_WITHOUT_ARG("--no-save-net", opt_set_invbool,
      &opt_save_net, "Don't save learnt changes"),
  OPT_WITH_ARG("--multi-tap=<n>", opt_set_uintval, opt_show_uintval,
      &opt_multi_tap, "read at n evenly spaced points in parallel"),
  OPT_WITHOUT_ARG("--use-multi-tap-path", opt_set_bool,
      &opt_use_multi_tap_path, "use multi-tap code path on single-tap tasks"),
  OPT_WITH_ARG("--momentum-style=<n>", opt_set_intval, opt_show_intval,
      &opt_momentum_style, "0: weighted, 1: Nesterov, 2: simplified N., 3: classical"),
  OPT_WITH_ARG("--init-weight-scale=<float>", opt_set_floatval, opt_show_floatval,
      &opt_init_weight_scale, "scale newly initialised weights (try ~1.0)"),
  OPT_WITH_ARG("--report-interval=<n>", opt_set_uintval_bi, opt_show_uintval_bi,
      &opt_report_interval, "how often to validate and report"),
  OPT_WITH_ARG("--confab-only=<chars>", opt_set_uintval, opt_show_uintval,
      &opt_confab_only, "no training, only confabulate this many characters"),
  OPT_WITH_ARG("--bottom-layer=<nodes>", opt_set_uintval, opt_show_uintval,
      &opt_bottom_layer, "use a bottom layer with this many output nodes"),
  OPT_WITH_ARG("--top-learn-rate-scale=<float>", opt_set_floatval, opt_show_floatval,
      &opt_top_learn_rate_scale, "top layer learn rate (relative)"),
  OPT_WITH_ARG("--bottom-learn-rate-scale=<float>", opt_set_floatval, opt_show_floatval,
      &opt_bottom_learn_rate_scale, "bottom layer learn rate (relative)"),
  OPT_WITH_ARG("--periodic-weight-noise=<stddev>", opt_set_floatval, opt_show_floatval,
      &opt_periodic_weight_noise, "periodically add this much gaussian noise to weights"),

  OPT_WITHOUT_ARG("-h|--help", opt_usage_and_exit,
      ": Rnn modelling of text at the character level",
      "Print this message."),
  OPT_ENDTABLE
};

static inline void
long_confab(RecurNN *net, int len, int rows, char *alphabet, float bias, int learn_caps){
  int i, j;
  char confab[len * rows + 1];
  confab[len * rows] = 0;
  confabulate(net, confab, len * rows, alphabet, bias, learn_caps);
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


static char*
construct_net_filename(void){
  char s[260];
  int alpha_size = strlen(opt_alphabet);
  int input_size = alpha_size + (opt_learn_capitals ? 1 : 0);
  int output_size = alpha_size + (opt_learn_capitals ? 2 : 0);
  snprintf(s, sizeof(s), "%s--%s", opt_alphabet, opt_collapse_chars);
  u32 sig = rnn_hash32(s);
  if (opt_bottom_layer){
    snprintf(s, sizeof(s), "%s-s%0" PRIx32 "-i%d-b%d-h%d-o%d-c%d.net", opt_basename,
        sig, input_size, opt_bottom_layer, opt_hidden_size, output_size,
        opt_learn_capitals);
  }
  else{
    snprintf(s, sizeof(s), "%s-s%0" PRIx32 "-i%d-h%d-o%d-c%d.net", opt_basename,
        sig, input_size, opt_hidden_size, output_size,
        opt_learn_capitals);
  }
  DEBUG("filename: %s", s);
  return strdup(s);
}

static inline int
bounded_init_method(int m){
  if (m > 0 && m < RNN_INIT_LAST){
    return m;
  }
  STDERR_DEBUG("ignoring bad init-method %d", m);
  return DEFAULT_INIT_METHOD;
}

#define IN_RANGE_01(x) ((x) >= 0.0f && (x) <= 1.0f)

static void
initialise_net(RecurNN *net){
  /*start off with a default set of parameters */
  struct RecurInitialisationParameters p;
  rnn_init_default_weight_parameters(net, &p);
  p.method = bounded_init_method(opt_init_method);
  p.submethod = bounded_init_method(opt_init_submethod);

  /*When the initialisation is using some fancy loop method, the top and
    possibly the bias and input weights use flat or fan-in initialisation.
    That means we need to set flat and fan-in parameters in any case.
  */
  if (opt_flat_init_distribution){
    p.flat_shape = opt_flat_init_distribution;
  }
  float variance = opt_init_variance;
  if (variance < 0){
    variance = RNN_INITIAL_WEIGHT_VARIANCE_FACTOR / net->h_size;
  }
  p.flat_variance = variance;
  p.flat_perforation = opt_perforate_weights;

  if (IN_RANGE_01(opt_init_input_probability)){
    p.run_input_probability = opt_init_input_probability;
  }
  if (opt_init_input_magnitude > 0){
    p.run_input_magnitude = opt_init_input_magnitude;
  }
  if (opt_init_hidden_gain > 0){
    p.run_gain = opt_init_hidden_gain;
  }
  if (opt_init_hidden_run_length > 0){
    p.run_len_mean = opt_init_hidden_run_length;
  }
  if (opt_init_hidden_run_deviation > 0){
    p.run_len_stddev = opt_init_hidden_run_deviation;
  }
  rnn_randomise_weights_clever(net, &p);

  if (opt_init_weight_scale > 0){
    rnn_scale_initial_weights(net, opt_init_weight_scale);
  }
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
    u32 flags = RNN_NET_FLAG_STANDARD;
    if (opt_bptt_adaptive_min){/*on by default*/
      flags |= RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR;
    }
    if(opt_bottom_layer){
      net = rnn_new_with_bottom_layer(input_size, opt_bottom_layer,
          opt_hidden_size, output_size, flags, opt_rng_seed,
          opt_logfile, opt_bptt_depth, opt_learn_rate,
          opt_momentum, 0);
    }
    else{
      net = rnn_new(input_size, opt_hidden_size,
          output_size, flags, opt_rng_seed,
          opt_logfile, opt_bptt_depth, opt_learn_rate,
          opt_momentum);
    }
    initialise_net(net);
    net->bptt->momentum_weight = opt_momentum_weight;
    if (opt_periodic_pgm_dump){
      rnn_multi_pgm_dump(net, PGM_DUMP_STRING, opt_basename);
    }
  }
  else if (opt_override){
    RecurNNBPTT *bptt = net->bptt;
    bptt->learn_rate = opt_learn_rate;
    bptt->momentum = opt_momentum;
    bptt->momentum_weight = opt_momentum_weight;
  }
  net->bptt->ho_scale = opt_top_learn_rate_scale;
  if (net->bottom_layer){
    net->bottom_layer->learn_rate_scale = opt_bottom_learn_rate_scale;
  }

  return net;
}


static inline void
finish(RnnCharModel *model, Ventropy *v){
  if (opt_filename && opt_save_net){
    rnn_save_net(model->net, opt_filename, 1);
  }
  BELOW_QUIET_LEVEL(3){
    RecurNNBPTT *bptt = model->net->bptt;
    float ventropy = rnn_char_calc_ventropy(model, v, 0);
    DEBUG("final entropy %.3f; learn rate %.2g; momentum %.2g",
        ventropy, bptt->learn_rate, bptt->momentum);
  }
  exit(0);
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
  RecurNN *net = load_or_create_net();
  if (opt_confab_only){
    long_confab(net, opt_confab_only, 1, opt_alphabet, opt_confab_bias, opt_learn_capitals);
    exit(0);
  }

  RnnCharModel model = {
    .net = net,
    .n_training_nets = MAX(opt_multi_tap, 1),

    .pgm_name = opt_basename,
    .batch_size = opt_batch_size,

    .momentum = opt_momentum,
    .momentum_soft_start = opt_momentum_soft_start,
    .momentum_style = opt_momentum_style,
    .learn_caps = opt_learn_capitals,
    .periodic_pgm_dump = opt_periodic_pgm_dump,
    .temporal_pgm_dump = opt_temporal_pgm_dump,
    .periodic_weight_noise = opt_periodic_weight_noise,
    .quiet = opt_quiet,
    .report_interval = opt_report_interval,
    .save_net = opt_save_net,
    .use_multi_tap_path = opt_use_multi_tap_path,
    .alphabet = opt_alphabet
  };

  if (model.n_training_nets > 1){
    model.training_nets = rnn_new_training_set(net, model.n_training_nets);
  }
  else{
    model.training_nets = &net;
  }

  RecurNN *confab_net = rnn_clone(net,
      net->flags & ~(RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS),
      RECUR_RNG_SUBSEED,
      NULL);
  RecurNN *validate_net = rnn_clone(net,
      net->flags & ~(RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS),
      RECUR_RNG_SUBSEED,
      NULL);

  init_schedule(&model.schedule, opt_learn_rate_inertia, opt_learn_rate_min,
      opt_learn_rate_scale);
  long len;
  u8* validate_text;
  u8* text = alloc_and_collapse_text(opt_textfile,
      opt_alphabet, (u8 *)opt_collapse_chars, &len, opt_learn_capitals,
      opt_quiet);
  if (opt_dump_collapsed_text){
    dump_collapsed_text(text, len, opt_dump_collapsed_text, opt_alphabet);
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
    model.input_ppm = temporal_ppm_alloc(net->i_size, 300, "input_layer", 0,
        PGM_DUMP_COLOUR, NULL);
    model.error_ppm = temporal_ppm_alloc(net->o_size, 300, "output_error", 0,
        PGM_DUMP_COLOUR, NULL);
  }
  Ventropy v;

  init_ventropy(&v, validate_net, validate_text,
      opt_validate_chars, opt_validation_overlap);

  if (opt_stop < 0){
    opt_stop = net->generation - opt_stop;
  }

  rnn_print_net_stats(net);

  int finished = 0;
  BELOW_QUIET_LEVEL(2){
    START_TIMER(run);
    for (int i = 0; finished == 0; i++){
      DEBUG("Starting epoch %d. learn rate %g.", i, net->bptt->learn_rate);
      START_TIMER(epoch);

      finished = epoch(&model, confab_net, &v,
          text, len, start_char, opt_stop, opt_confab_bias, CONFAB_SIZE, opt_quiet);
      DEBUG_TIMER(epoch);
      DEBUG_TIMER(run);
      start_char = 0;
    }
  }
  else {/* quiet level 2+ */
    for (;finished == 0;){
      finished = epoch(&model, NULL, &v,
          text, len, start_char, opt_stop, 0, 0, opt_quiet);
      start_char = 0;
    }
  }
  if (finished){
    finish(&model, &v);
  }

  free(text);
  if (opt_multi_tap < 2){
    rnn_delete_net(net);
  }
  else {
    rnn_delete_training_set(model.training_nets, opt_multi_tap, 0);
  }
  rnn_delete_net(confab_net);
  rnn_delete_net(validate_net);

  temporal_ppm_free(model.input_ppm);
  temporal_ppm_free(model.error_ppm);
}
