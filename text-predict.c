/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> GPL2+

This trains a network to predict the next character in a text sequence.

Unlike most of the Recur repository, this file is licensed under the GNU
General Public License, version 2 or greater. That is because it is linked to
ccan/opt which is also GPL2+.

Because of ccan/opt, --help will tell you something.
*/

#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include <math.h>
#include "path.h"
#include "badmaths.h"
#include <errno.h>
#include <stdio.h>
#include <fenv.h>
#include <ctype.h>
#include "charmodel.h"
#include "utf8.h"
#include "opt-helpers.h"
#include <limits.h>

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

#define DEFAULT_PGM_DUMP_IMAGES "ihw how"
#define DEFAULT_PERIODIC_PGM_DUMP 0
#define DEFAULT_TEMPORAL_PGM_DUMP 0
#define DEFAULT_RELOAD 0
#define DEFAULT_LEARN_RATE 0.001
#define DEFAULT_LEARN_RATE_MIN 0
#define DEFAULT_LEARN_RATE_INERTIA 0
#define DEFAULT_LEARN_RATE_SCALE 0.5
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
#define DEFAULT_LOG_FILE "text.log"
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
#define DEFAULT_ACTIVATION RNN_RELU

#define DEFAULT_FORCE_METADATA 0
#define DEFAULT_DUMP_COLLAPSED_TEXT NULL
#define DEFAULT_MULTI_TAP 0
#define DEFAULT_USE_MULTI_TAP_PATH 0
#define DEFAULT_LEARNING_STYLE RNN_MOMENTUM_WEIGHTED
#define DEFAULT_INIT_WEIGHT_SCALE 0
#define DEFAULT_REPORT_INTERVAL 1024
#define DEFAULT_BOTTOM_LAYER 0
#define DEFAULT_TOP_LEARN_RATE_SCALE 1.0f
#define DEFAULT_BOTTOM_LEARN_RATE_SCALE 1.0f
#define DEFAULT_PERIODIC_WEIGHT_NOISE 0
#define DEFAULT_CASE_INSENSITIVE 1
#define DEFAULT_UTF8 0
#define DEFAULT_COLLAPSE_SPACE 1
#define DEFAULT_FIND_ALPHABET_THRESHOLD 0
#define DEFAULT_FIND_ALPHABET_DIGIT_ADJUST 1.0
#define DEFAULT_FIND_ALPHABET_ALPHA_ADJUST 1.0
#define DEFAULT_PRESYNAPTIC_NOISE 0.0f
#define DEFAULT_ADJUST_NOISE false
#define DEFAULT_ADAGRAD_BALLAST 200.0f
#define DEFAULT_ADADELTA_BALLAST 0
#define DEFAULT_FP_EXCEPTION_LEVEL 1

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
static char * opt_logfile = NULL;
static char * opt_basename = DEFAULT_BASENAME;
static char * opt_alphabet = NULL;
static char * opt_collapse_chars = DEFAULT_COLLAPSE_CHARS;
static char * opt_textfile = DEFAULT_TEXT;
static char * opt_dump_collapsed_text = DEFAULT_DUMP_COLLAPSED_TEXT;
static bool opt_reload = DEFAULT_RELOAD;
static float opt_momentum_weight = DEFAULT_MOMENTUM_WEIGHT;
static float opt_momentum_soft_start = DEFAULT_MOMENTUM_SOFT_START;
static s64 opt_rng_seed = DEFAULT_RNG_SEED;
static int opt_stop = DEFAULT_STOP;
static int opt_validate_chars = DEFAULT_VALIDATE_CHARS;
static int opt_validation_overlap = DEFAULT_VALIDATION_OVERLAP;
static int opt_start_char = DEFAULT_START_CHAR;
static bool opt_override = DEFAULT_OVERRIDE;
static bool opt_force_metadata = DEFAULT_FORCE_METADATA;
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
static bool opt_periodic_pgm_dump = DEFAULT_PERIODIC_PGM_DUMP;
static bool opt_temporal_pgm_dump = DEFAULT_TEMPORAL_PGM_DUMP;
static char *opt_pgm_dump_images = DEFAULT_PGM_DUMP_IMAGES;
static char **orig_pgm_dump_images = &opt_pgm_dump_images;
static float opt_confab_bias = DEFAULT_CONFAB_BIAS;
static bool opt_save_net = DEFAULT_SAVE_NET;
static uint opt_multi_tap = DEFAULT_MULTI_TAP;
static bool opt_use_multi_tap_path = DEFAULT_USE_MULTI_TAP_PATH;
static int opt_learning_style = DEFAULT_LEARNING_STYLE;
static uint opt_report_interval = DEFAULT_REPORT_INTERVAL;
static uint opt_bottom_layer = DEFAULT_BOTTOM_LAYER;
static float opt_top_learn_rate_scale = DEFAULT_TOP_LEARN_RATE_SCALE;
static float opt_bottom_learn_rate_scale = DEFAULT_BOTTOM_LEARN_RATE_SCALE;
static float opt_periodic_weight_noise = DEFAULT_PERIODIC_WEIGHT_NOISE;
static bool opt_case_insensitive = DEFAULT_CASE_INSENSITIVE;
static bool opt_utf8 = DEFAULT_UTF8;
static bool opt_collapse_space = DEFAULT_COLLAPSE_SPACE;
static double opt_find_alphabet_threshold = DEFAULT_FIND_ALPHABET_THRESHOLD;
static double opt_find_alphabet_digit_adjust = DEFAULT_FIND_ALPHABET_DIGIT_ADJUST;
static double opt_find_alphabet_alpha_adjust = DEFAULT_FIND_ALPHABET_ALPHA_ADJUST;
static float opt_presynaptic_noise = DEFAULT_PRESYNAPTIC_NOISE;
static bool opt_adjust_noise = DEFAULT_ADJUST_NOISE;
static float opt_ada_ballast = -1;
static int opt_activation = DEFAULT_ACTIVATION;
static int opt_fp_exception_level = DEFAULT_FP_EXCEPTION_LEVEL;
static uint opt_diagonal_only_section = 0;
static float opt_diagonal_only_boost = 0;
static uint opt_diagonal_only_friends = 0;

static struct opt_table options[] = {
  OPT_WITH_ARG("-H|--hidden-size=<n>", opt_set_uintval, opt_show_uintval,
      &opt_hidden_size, "number of hidden nodes"),
  OPT_WITH_ARG("-d|--depth=<n>", opt_set_uintval, opt_show_uintval,
      &opt_bptt_depth, "max depth of BPTT recursion"),
  /* Some machines quibble over long vs long long when both are 64 bit, so we
     need to appease them without upsetting the others. This is not
     necessarily right, but seems to work. */
#if LONG_BIT == 64
  OPT_WITH_ARG("-r|--rng-seed=<seed>", opt_set_longval_bi, opt_show_longval_bi,
      &opt_rng_seed, "RNG seed (-1 for auto)"),
#else
  OPT_WITH_ARG("-r|--rng-seed=<seed>", opt_set_longlongval_bi,
      opt_show_longlongval_bi,
      &opt_rng_seed, "RNG seed (-1 for auto)"),
#endif
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
  OPT_WITH_ARG("-l|--learn-rate=<0-1>", opt_set_floatval, opt_show_floatval,
      &opt_learn_rate, "initial learning rate"),
  OPT_WITH_ARG("--learn-rate-min=<0-1>", opt_set_floatval01, opt_show_floatval,
      &opt_learn_rate_min, "minimum learning rate (>learn-rate is off)"),
  OPT_WITH_ARG("--learn-rate-inertia=<int>", opt_set_intval, opt_show_intval,
      &opt_learn_rate_inertia, "tardiness of learn-rate reduction (try 30-90)"),
  OPT_WITH_ARG("--learn-rate-scale=<int>", opt_set_floatval, opt_show_floatval,
      &opt_learn_rate_scale, "size of learn rate reductions"),
  OPT_WITH_ARG("-m|--momentum=<0-1>", opt_set_floatval01, opt_show_floatval,
      &opt_momentum, "momentum (or decay rate with adadelta)"),
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
  OPT_WITHOUT_ARG("--force-metadata", opt_set_bool,
      &opt_force_metadata, "force loading of net in face of metadata mismatch"),
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
  OPT_WITH_ARG("--periodic-pgm-dump-images", opt_set_charp, opt_show_charp,
      &opt_pgm_dump_images, "which images to dump ({ih,ho,bi}[wmdt])*"),
  OPT_WITH_ARG("--confab-bias", opt_set_floatval, opt_show_floatval,
      &opt_confab_bias, "bias toward probable characters in confab "
      "(100 == deterministic)"),
  OPT_WITHOUT_ARG("--no-save-net", opt_set_invbool,
      &opt_save_net, "Don't save learnt changes"),
  OPT_WITH_ARG("--multi-tap=<n>", opt_set_uintval, opt_show_uintval,
      &opt_multi_tap, "read at n evenly spaced points in parallel"),
  OPT_WITHOUT_ARG("--use-multi-tap-path", opt_set_bool,
      &opt_use_multi_tap_path, "use multi-tap code path on single-tap tasks"),
  OPT_WITH_ARG("--learning-style=<n>", opt_set_intval, opt_show_intval,
      &opt_learning_style, "0: weighted, 1: Nesterov, 2: simplified N., "
      "3: classical, 4: adagrad, 5: adadelta, 6: rprop"),
  OPT_WITH_ARG("--init-weight-scale=<float>", opt_set_floatval, opt_show_floatval,
      &opt_init_weight_scale, "scale newly initialised weights (try ~1.0)"),
  OPT_WITH_ARG("--report-interval=<n>", opt_set_uintval_bi, opt_show_uintval_bi,
      &opt_report_interval, "how often to validate and report"),
  OPT_WITH_ARG("--bottom-layer=<nodes>", opt_set_uintval, opt_show_uintval,
      &opt_bottom_layer, "use a bottom layer with this many output nodes"),
  OPT_WITH_ARG("--top-learn-rate-scale=<float>", opt_set_floatval, opt_show_floatval,
      &opt_top_learn_rate_scale, "top layer learn rate (relative)"),
  OPT_WITH_ARG("--bottom-learn-rate-scale=<float>", opt_set_floatval, opt_show_floatval,
      &opt_bottom_learn_rate_scale, "bottom layer learn rate (relative)"),
  OPT_WITH_ARG("--periodic-weight-noise=<stddev>", opt_set_floatval, opt_show_floatval,
      &opt_periodic_weight_noise, "periodically add this much gaussian noise to weights"),
  OPT_WITHOUT_ARG("--case-sensitive", opt_set_invbool,
      &opt_case_insensitive, "Treat capitals as their separate symbols"),
  OPT_WITHOUT_ARG("--case-insensitive", opt_set_bool,
      &opt_case_insensitive, "Treat capitals as lower case characters (ASCII only)"),
  OPT_WITHOUT_ARG("--utf8", opt_set_bool,
      &opt_utf8, "Parse text as UTF8"),
  OPT_WITHOUT_ARG("--no-utf8", opt_set_invbool,
      &opt_utf8, "Parse text as 8 bit symbols"),
  OPT_WITHOUT_ARG("--adjust-noise", opt_set_bool,
      &opt_adjust_noise, "Decay presynaptic and weight noise with learn-rate"),
  OPT_WITHOUT_ARG("--collapse-space", opt_set_bool,
      &opt_collapse_space, "Runs of whitespace collapse to single space"),
  OPT_WITHOUT_ARG("--no-collapse-space", opt_set_invbool,
      &opt_collapse_space, "Predict whitespace characters individually"),
  OPT_WITH_ARG("--find-alphabet-threshold", opt_set_doubleval, opt_show_doubleval,
      &opt_find_alphabet_threshold, "minimum frequency for character to be included"),
  OPT_WITH_ARG("--find-alphabet-digit-adjust", opt_set_doubleval, opt_show_doubleval,
      &opt_find_alphabet_digit_adjust, "adjust digit frequency for alphabet calculations"),
  OPT_WITH_ARG("--find-alphabet-alpha-adjust", opt_set_doubleval, opt_show_doubleval,
      &opt_find_alphabet_alpha_adjust, "adjust letter frequency for alphabet calculation"),
  OPT_WITH_ARG("--presynaptic-noise", opt_set_floatval, opt_show_floatval,
      &opt_presynaptic_noise, "deviation of noise to add before non-linear transform"),
  OPT_WITH_ARG("--ada-ballast", opt_set_floatval, opt_show_floatval,
      &opt_ada_ballast, "adagrad/adadelta accumulators start at this value"),
  OPT_WITH_ARG("--activation", opt_set_intval, opt_show_intval,
      &opt_activation, "1: ReLU, 2: ReSQRT, 5: clipped ReLU"),
  OPT_WITH_ARG("--fp-exception-level", opt_set_intval, opt_show_intval,
      &opt_fp_exception_level, "floating point exceptions; 0: none, 1: some, 2: all"),
  OPT_WITH_ARG("--diagonal-only-section", opt_set_uintval, opt_show_uintval,
      &opt_diagonal_only_section, "restrict this many hidden neurons to diagonal "
      "connections"),
  OPT_WITH_ARG("--diagonal-only-friends", opt_set_uintval, opt_show_uintval,
      &opt_diagonal_only_friends, "this many of the diagonal-only neurons have "
      "a recurrent friend"),
  OPT_WITH_ARG("--diagonal-only-boost", opt_set_floatval, opt_show_floatval,
      &opt_diagonal_only_boost, "add to weights in --diagonal-only-section"),

  OPT_WITHOUT_ARG("-h|--help", opt_usage_and_exit,
      ": Rnn modelling of text at the character level",
      "Print this message."),
  OPT_ENDTABLE
};

static inline int
bounded_init_method(int m){
  if (m > 0 && m < RNN_INIT_LAST){
    return m;
  }
  STDERR_DEBUG("ignoring bad init-method %d", m);
  return DEFAULT_INIT_METHOD;
}

#define SET_IF_POSITIVE(a, b) (a) = ((b) > 0 ? (b) : (a))

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
  SET_IF_POSITIVE(p.run_input_magnitude, opt_init_input_magnitude);
  SET_IF_POSITIVE(p.run_gain, opt_init_hidden_gain);
  SET_IF_POSITIVE(p.run_len_mean, opt_init_hidden_run_length);
  SET_IF_POSITIVE(p.run_len_stddev, opt_init_hidden_run_deviation);

  rnn_randomise_weights_clever(net, &p);

  if (opt_init_weight_scale > 0){
    rnn_scale_initial_weights(net, opt_init_weight_scale);
  }
}

static void
exit_unless_metadata_matches(RecurNN *net, struct RnnCharMetadata *m,
    bool trust_file_metadata, bool force_metadata){
  int r = rnn_char_check_metadata(net, m, trust_file_metadata,
      force_metadata);
  if (r){
    if (r == -2){
      DEBUG("Aborting. (use --force-metadata to ignore metadata issues)");
    }
    else {
      DEBUG("I cannot carry on!");
    }
    exit(1);
  }
}


static RecurNN *
create_net(const char *filename, int alpha_len)
{
  RecurNN *net;
  u32 flags = RNN_NET_FLAG_STANDARD;
  if (opt_bptt_adaptive_min){/*on by default*/
    flags |= RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR;
  }
  if (opt_learning_style == RNN_ADADELTA || opt_learning_style == RNN_RPROP){
    flags |= RNN_NET_FLAG_AUX_ARRAYS;
  }
  net = rnn_new_with_bottom_layer(alpha_len, opt_bottom_layer,
      opt_hidden_size, alpha_len, flags, opt_rng_seed,
      opt_logfile, opt_bptt_depth, opt_learn_rate,
      opt_momentum, opt_presynaptic_noise, opt_activation, 0);
  initialise_net(net);
  net->bptt->momentum_weight = opt_momentum_weight;

  net->bptt->ho_scale = opt_top_learn_rate_scale;
  if (net->bottom_layer){
    net->bottom_layer->learn_rate_scale = opt_bottom_learn_rate_scale;
  }
  return net;
}


static inline void
finish(RnnCharModel *model, RnnCharVentropy *v){
  if (model->filename && opt_save_net){
    rnn_save_net(model->net, model->filename, 1);
  }
  BELOW_QUIET_LEVEL(3){
    RecurNNBPTT *bptt = model->net->bptt;
    float ventropy = rnn_char_calc_ventropy(model, v, 0);
    DEBUG("final entropy %.3f; learn rate %.2g; momentum %.2g",
        ventropy, bptt->learn_rate, bptt->momentum);
  }
}

static void prepare_diagonal_only_section(RecurNN *net,
    uint len, uint friends, float boost)
{
  if (len > (unsigned)net->hidden_size){
    FATAL_ERROR("diagonal_only_section is too big! %u > %d",
        len, net->hidden_size);
  }
  rnn_clear_diagonal_only_section(net, len, friends);

  int h_end = net->hidden_size + 1;
  int start = h_end - len;
  int stop = h_end;
  if (boost){
    for (int i = start; i < stop; i++){
      net->ih_weights[i * net->h_size + i] += boost;
    }
  }
}

static void
load_and_train_model(struct RnnCharMetadata *m, RnnCharAlphabet *alphabet)
{
  char *metadata = NULL;
  RnnCharModel model = {
    .n_training_nets = MAX(opt_multi_tap, 1),
    .batch_size = opt_batch_size,
    .momentum = opt_momentum,
    .momentum_soft_start = opt_momentum_soft_start,
    .learning_style = opt_learning_style,
    .periodic_weight_noise = opt_periodic_weight_noise,
    .report_interval = opt_report_interval,
    .save_net = opt_save_net,
    .use_multi_tap_path = opt_use_multi_tap_path,
    .alphabet = alphabet,
    .images = {
      .basename = opt_basename,
      .temporal_pgm_dump = opt_temporal_pgm_dump
    }
  };
  if (opt_periodic_pgm_dump || opt_pgm_dump_images != *orig_pgm_dump_images){
    model.images.periodic_pgm_dump_string = opt_pgm_dump_images;
  }

  if (opt_filename){
    model.filename = opt_filename;
  }
  else{
    model.filename = rnn_char_construct_net_filename(m, opt_basename, alphabet->len,
        opt_bottom_layer, opt_hidden_size, alphabet->len);
  }

  RecurNN *net = opt_reload ? rnn_load_net(model.filename) : NULL;
  if (net){
    bool trust_metadata = (bool)opt_filename;
    exit_unless_metadata_matches(net, m, trust_metadata, opt_force_metadata);
  }
  else {
    DEBUG("Could not load '%s', let's make a new net", model.filename);
    net = create_net(model.filename, alphabet->len);
    metadata = rnn_char_construct_metadata(m);
    net->metadata = metadata;
  }

  rnn_set_log_file(net, opt_logfile, 1);
  if (opt_override){
    RecurNNBPTT *bptt = net->bptt;
    bptt->learn_rate = opt_learn_rate;
    bptt->momentum = opt_momentum;
    bptt->momentum_weight = opt_momentum_weight;
  }

  model.net = net;
  model.training_nets = rnn_new_training_set(net, model.n_training_nets);

  if (model.images.temporal_pgm_dump){
    model.images.input_ppm = temporal_ppm_alloc(net->i_size, 300, "input_layer", 0,
        PGM_DUMP_COLOUR, NULL);
    model.images.error_ppm = temporal_ppm_alloc(net->o_size, 300, "output_error", 0,
        PGM_DUMP_COLOUR, NULL);
  }

  RecurNN *confab_net = rnn_clone(net,
      net->flags & ~(RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS),
      RECUR_RNG_SUBSEED,
      NULL);
  RecurNN *validate_net = rnn_clone(net,
      net->flags & ~(RNN_NET_FLAG_OWN_BPTT | RNN_NET_FLAG_OWN_WEIGHTS),
      RECUR_RNG_SUBSEED,
      NULL);

  rnn_char_init_schedule(&model.schedule, opt_learn_rate_inertia, opt_learn_rate_min,
      opt_learn_rate_scale, opt_adjust_noise);

  if (model.learning_style == RNN_ADAGRAD){
    if (opt_ada_ballast < 0){
      opt_ada_ballast = DEFAULT_ADAGRAD_BALLAST;
    }
    rnn_set_momentum_values(net, opt_ada_ballast);
  }
  else if (model.learning_style == RNN_ADADELTA){
    if (opt_ada_ballast < 0){
      opt_ada_ballast = DEFAULT_ADADELTA_BALLAST;
    }
    rnn_set_momentum_values(net, opt_ada_ballast);
  }
  else if (model.learning_style == RNN_RPROP){
    rnn_set_aux_values(net, 1);
  }
  /* get text and validation text */
  int text_len;
  u8* validate_text;

  u8* text = rnn_char_load_new_encoded_text(opt_textfile, alphabet,
      &text_len, opt_quiet);
  if (opt_dump_collapsed_text){
    rnn_char_dump_collapsed_text(text, text_len, opt_dump_collapsed_text, m->alphabet);
  }

  if (opt_validate_chars > 2 &&
      text_len - opt_validate_chars > 2){
    text_len -= opt_validate_chars;
    validate_text = text + text_len;
  }
  else {
    if (opt_validate_chars){
      DEBUG("--validate-chars is too small or too big (%d)"
          " and will be ignored", opt_validate_chars);
      opt_validate_chars = 0;
    }
    validate_text = NULL;
  }
  RnnCharVentropy v;

  rnn_char_init_ventropy(&v, validate_net, validate_text,
      opt_validate_chars, opt_validation_overlap);


  /*start_char can only go up to text_len - 1, because the i + 1th character is the
    one being predicted, hence has to be accessed for feedback. */
  int start_char;
  if (opt_start_char >= 0 && opt_start_char < text_len - 1){
    start_char = opt_start_char;
  }
  else {
    start_char = net->generation % (text_len - 1);
  }

  if (opt_stop < 0){
    opt_stop = net->generation - opt_stop;
  }

  rnn_print_net_stats(net);

  int confab_line_end = -1;
  if (opt_collapse_space == 0){
    confab_line_end = rnn_char_get_codepoint(alphabet, "\n");
  }

  if (opt_diagonal_only_section){
    prepare_diagonal_only_section(net, opt_diagonal_only_section,
        opt_diagonal_only_friends,
        opt_diagonal_only_boost);
  }

  if (model.images.periodic_pgm_dump_string){
    rnn_multi_pgm_dump(net, model.images.periodic_pgm_dump_string,
        model.images.basename);
  }

  int finished = 0;
  BELOW_QUIET_LEVEL(2){
    START_TIMER(run);
    for (int i = 0; ! finished; i++){
      DEBUG("Starting epoch %d. learn rate %g. collapse space %d",
          i + 1, net->bptt->learn_rate, opt_collapse_space);
      START_TIMER(epoch);
      finished = rnn_char_epoch(&model, confab_net, &v,
          text, text_len, start_char, opt_stop, opt_confab_bias, CONFAB_SIZE,
          confab_line_end, opt_quiet, opt_diagonal_only_section,
          opt_diagonal_only_friends);
      DEBUG_TIMER(epoch);
      DEBUG_TIMER(run);
      start_char = 0;
    }
  }
  else {/* quiet level 2+ */
    do {
      finished = rnn_char_epoch(&model, NULL, &v,
          text, text_len, start_char, opt_stop, 0, 0,
          confab_line_end, opt_quiet, opt_diagonal_only_section,
          opt_diagonal_only_friends);
      start_char = 0;
    }
    while (! finished);
  }
  if (finished){
    finish(&model, &v);
  }

  free(text);

  rnn_delete_training_set(model.training_nets, model.n_training_nets, 0);
  rnn_delete_net(confab_net);
  rnn_delete_net(validate_net);
  rnn_char_delete_ventropy(&v);

  if (model.images.input_ppm){
    temporal_ppm_free(model.images.input_ppm);
  }
  if (model.images.error_ppm){
    temporal_ppm_free(model.images.error_ppm);
  }
  if (model.filename != opt_filename) {
    free(model.filename);
  }
  rnn_char_free_alphabet(alphabet);
  if (metadata != NULL) {
    free(metadata);
  }
}

void train_new_or_existing_model(void){
  /*find an alphabet, somehow, because it will affect the filename, which will
    be necessary to attempt to load.

    This may involve reading the training text an extra time.
  */
  bool locally_allocated_alphabet = false;
  RnnCharAlphabet *alphabet = rnn_char_new_alphabet();
  rnn_char_alphabet_set_flags(alphabet,
      opt_case_insensitive,
      opt_utf8,
      opt_collapse_space);

  if (opt_find_alphabet_threshold && ! opt_alphabet){
    DEBUG("Looking for alphabet with threshold %f", opt_find_alphabet_threshold);
    int raw_text_len;
    char* text;
    int err = rnn_char_alloc_file_contents(opt_textfile, &text, &raw_text_len);
    if (err){
      DEBUG("Couldn't read text file '%s'. Goodbye", opt_textfile);
      exit(1);
    }
    rnn_char_find_alphabet_s(text, raw_text_len, alphabet,
        opt_find_alphabet_threshold,
        opt_find_alphabet_digit_adjust,
        opt_find_alphabet_alpha_adjust);

    free(text);
    if (alphabet->len < 1){
      DEBUG("Trouble finding an alphabet");
      exit(1);
    }
    opt_alphabet = new_string_from_codepoints(alphabet->points, alphabet->len, opt_utf8);
    opt_collapse_chars = new_string_from_codepoints(alphabet->collapsed_points,
        alphabet->collapsed_len, opt_utf8);
    locally_allocated_alphabet = true;
  }
  else { /*use given or default alphabet */
    if (! opt_alphabet){
      opt_alphabet = DEFAULT_CHARSET;
    }
    alphabet->len = fill_codepoints_from_string(alphabet->points, 256,
        opt_alphabet, opt_utf8);
    alphabet->collapsed_len = fill_codepoints_from_string(alphabet->collapsed_points,
        256, opt_alphabet, opt_utf8);
  }
  STDERR_DEBUG("Using alphabet of length %d: '%s'", alphabet->len, opt_alphabet);
  STDERR_DEBUG("collapsing these %d characters into first alphabet character: '%s'",
      alphabet->collapsed_len, opt_collapse_chars);

  struct RnnCharMetadata m = {
    .alphabet = opt_alphabet,
    .collapse_chars = opt_collapse_chars,
    .utf8 = opt_utf8,
    .case_insensitive = opt_case_insensitive,
    .collapse_space = opt_collapse_space
  };
  load_and_train_model(&m, alphabet);
  if (locally_allocated_alphabet) {
    free(opt_alphabet);
    free(opt_collapse_chars);
  }
}


int
main(int argc, char *argv[]){
  char *mem = NULL;
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

  /*Default floating point exception level is 1,
    catching only bad FP issues */
  switch(opt_fp_exception_level){
  case 0:
    feclearexcept(FE_ALL_EXCEPT);
    break;
  case 1:
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    break;
  case 2:
    feenableexcept(FE_ALL_EXCEPT);
  }

  /* make sure there is some logfile */
  if (! opt_logfile){
    if (opt_basename){
      int n = asprintf(&opt_logfile, "%s.log", opt_basename);
      mem = opt_logfile;
      if (n < 5){
        FATAL_ERROR("error setting log filename from basename");
      }
    }
    else {
      opt_logfile = DEFAULT_LOG_FILE;
    }
  }
  train_new_or_existing_model();
  opt_free_table();
  if (mem != NULL) {
    free(mem);
  }
}
