/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> GPL2+

Try to learn the class of each document.

Unlike most of the Recur repository, this file is licensed under the GNU
General Public License, version 2 or greater. That is because it is linked to
ccan/opt which is also GPL2+.

Because of ccan/opt, --help will tell you something.
*/

#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include "path.h"
#include "badmaths.h"
#include <errno.h>
#include <stdio.h>
#include <fenv.h>
#include <ctype.h>
#include "charmodel.h"
#include "utf8.h"
#include "colour.h"
#include "opt-helpers.h"

#define UNCLASSIFIED "*unclassified*"

/*XXX copied from xml-lang-classify, should merge */
static inline ALWAYS_INLINE int
lookup_class(char** class_lut, const char *class, int set_if_not_found){
  if (strcmp(class, UNCLASSIFIED)){
    int i;
    for (i = 0; class_lut[i] && i < 256; i++){
      if (strcmp(class, class_lut[i]) == 0){
        return i;
      }
    }
    if (set_if_not_found && i < 256){
      class_lut[i] = strndup(class, 1000);
      return i;
    }
  }
  return NO_CLASS;
}


static void
free_langblocks(RnnCharClassBlock *b){
  RnnCharClassBlock *next;
  for (; b; b = next){
    next = b->next;
    free((char *)b->text);
    free(b);
  }
}

static char*
new_full_text_from_blocks(RnnCharClassBlock *b, int *len){
  int cumlen = 0;
  int size = 1000 * 1000;
  char *text = malloc(size);
  for(; b; b = b->next){
    if (cumlen + b->len > size){
      size *= 2;
      text = realloc(text, size);
    }
    memcpy(text + cumlen, b->text, b->len);
    cumlen += b->len;
  }
  text = realloc(text, cumlen + 1);
  text[cumlen] = 0;
  *len = cumlen;
  return text;
}

static RnnCharAlphabet *
new_alphabet_from_blocks(RnnCharClassBlock *b, double alpha_threshold,
    double digit_adjust, double alpha_adjust, u32 flags){
  RnnCharAlphabet *alphabet = rnn_char_new_alphabet();
  alphabet->flags = flags;
  int len;
  char *fulltext = new_full_text_from_blocks(b, &len);
  int r = rnn_char_find_alphabet_s(fulltext, len, alphabet,
      alpha_threshold, digit_adjust, alpha_adjust);
  free(fulltext);
  if (r){
    DEBUG("could not find alphabet (error %d)", r);
    free(alphabet);
    return NULL;
  }
  return alphabet;
}


static RnnCharClassBlock *
read_class_blocks(char *filename, char *basedir, char **classes, int add_to_classes){
  FILE *f = fopen(filename, "r");
  if (!f){
    goto error;
  }
  RnnCharClassBlock *blocks = NULL;
  RnnCharClassBlock *prev = NULL;
  RnnCharClassBlock *b = NULL;
  char line[501];
  while (fgets(line, 500, f)){
    char *text;
    int len;
    char *fn = strtok(line, " \t");
    int err;
    if (basedir){
      char ffn[1001];
      snprintf(ffn, 1000, "%s/%s", basedir, fn);
      err = rnn_char_alloc_file_contents(ffn, &text, &len);
    }
    else{
      err = rnn_char_alloc_file_contents(fn, &text, &len);
    }
    if (err){
      fclose(f);
      goto error;
    }
    char *class = strtok(NULL, "\n");
    int class_code = lookup_class(classes, class, add_to_classes);
    prev = b;
    b = malloc(sizeof(RnnCharClassBlock));
    b->class_name = classes[class_code]; /*use the strduped copy, not the local buffer.*/
    b->class_code = class_code;
    b->text = text;
    b->len = len;
    b->next = NULL;
    if (prev){
      prev->next = b;
    }
    else{
      blocks = b;
    }
  }
  fclose(f);
  return blocks;
 error:
  return NULL;
}

static RnnCharClassifiedText *
new_charmodel_from_filelist(char *filename, char *basedir, char *validation_file,
    double alpha_threshold, double digit_adjust, double alpha_adjust, u32 flags,
    int ignore_start)
{
  char **classes = calloc(256, sizeof(char*));
  classes[NO_CLASS] = UNCLASSIFIED;
  RnnCharClassBlock *training_blocks = NULL;
  RnnCharClassBlock *validation_blocks = NULL;

  training_blocks = read_class_blocks(filename, basedir, classes, 1);
  if (validation_file){
    validation_blocks = read_class_blocks(validation_file, basedir, classes, 0);
  }

  RnnCharAlphabet *alphabet = new_alphabet_from_blocks(training_blocks, alpha_threshold,
      digit_adjust, alpha_adjust, flags);
  if (alphabet == NULL){
    goto error;
  }

  RnnCharClassifiedText *t = malloc(sizeof(*t));
  t->text = rnn_char_alloc_classified_text(training_blocks,
      alphabet, &t->len, ignore_start);
  if (validation_blocks){
    t->validation_text = rnn_char_alloc_classified_text(validation_blocks,
        alphabet, &t->validation_len, ignore_start);
  }
  else {
    t->validation_text = NULL;
    t->validation_len = 0;
  }

  t->alphabet = alphabet;
  t->lag = 0;

  int n;
  for (n = 0; n < 256; n++){
    if (classes[n] == NULL){
      break;
    }
  }
  t->n_classes = n;
  t->classes = classes;
  free_langblocks(training_blocks);
  free_langblocks(validation_blocks);
  return t;
 error:
  DEBUG("error in reading filelist!");
  free_langblocks(training_blocks);
  free_langblocks(validation_blocks);
  free(classes);
  return NULL;
}

#define DEFAULT_ADAGRAD_BALLAST 200.0f
#define DEFAULT_ADADELTA_BALLAST 0

static char *opt_classification_file = NULL;
static char *opt_validation_file = NULL;
static char *opt_classification_dir = NULL;
static double opt_alpha_threshold = 1e-4;
static double opt_alpha_adjust = 3.0;
static double opt_digit_adjust = 1.0;
static uint opt_ignore_start = 0;
static int opt_verbose = 0;
static uint opt_multi_tap = 20;
static uint opt_hidden_size = 199;
static u64 opt_rng_seed = 11;
static char *opt_logfile = NULL;
static uint opt_bptt_depth = 40;
static float opt_learn_rate = 0.001;
static float opt_momentum = 0.93;
static char *opt_basename = "text-classify";
static float opt_presynaptic_noise = 0;
static float opt_ada_ballast = -1;
static int opt_activation = RNN_RELU;
static int opt_learning_style = RNN_MOMENTUM_WEIGHTED;
static bool opt_save_net = true;
static int opt_epochs = 0;
static char *opt_filename = NULL;


static struct opt_table options[] = {
  OPT_WITHOUT_ARG("-h|--help", opt_usage_and_exit,
      ": Rnn classification of text at the character level",
      "Print this message."),

  OPT_WITH_ARG("-H|--hidden-size=<n>", opt_set_uintval, opt_show_uintval,
      &opt_hidden_size, "number of hidden nodes"),
  OPT_WITH_ARG("-r|--rng-seed=<seed>", opt_set_ulongval_bi, opt_show_ulongval_bi,
      &opt_rng_seed, "RNG seed (-1 for auto)"),
  OPT_WITH_ARG("--find-alphabet-threshold", opt_set_doubleval, opt_show_doubleval,
      &opt_alpha_threshold, "minimum frequency for character to be included"),
  OPT_WITH_ARG("--find-alphabet-digit-adjust", opt_set_doubleval, opt_show_doubleval,
      &opt_digit_adjust, "adjust digit frequency for alphabet calculations"),
  OPT_WITH_ARG("--find-alphabet-alpha-adjust", opt_set_doubleval, opt_show_doubleval,
      &opt_alpha_adjust, "adjust letter frequency for alphabet calculation"),
  OPT_WITH_ARG("-i|--ignore-start", opt_set_uintval, opt_show_uintval,
      &opt_ignore_start, "don't classify this many first characters per block"),
  OPT_WITHOUT_ARG("-v|--verbose", opt_inc_intval,
      &opt_verbose, "More debugging noise, if possible"),
  OPT_WITHOUT_ARG("-q|--quiet", opt_inc_intval,
      &opt_verbose, "Less debugging noise."),
  OPT_WITH_ARG("--multi-tap=<n>", opt_set_uintval, opt_show_uintval,
      &opt_multi_tap, "read at n evenly spaced points in parallel"),
  OPT_WITH_ARG("--log-file=<file>", opt_set_charp, opt_show_charp, &opt_logfile,
      "log to this filename"),
  OPT_WITH_ARG("-d|--depth=<n>", opt_set_uintval, opt_show_uintval,
      &opt_bptt_depth, "max depth of BPTT recursion"),
  OPT_WITH_ARG("-m|--momentum=<0-1>", opt_set_floatval01, opt_show_floatval,
      &opt_momentum, "momentum"),
  OPT_WITH_ARG("-l|--learn-rate=<0-1>", opt_set_floatval01, opt_show_floatval,
      &opt_learn_rate, "initial learning rate"),
  OPT_WITH_ARG("-n|--basename=<tag>", opt_set_charp, opt_show_charp, &opt_basename,
      "construct log, image, net filenames from this root"),
  OPT_WITH_ARG("--presynaptic-noise", opt_set_floatval, opt_show_floatval,
      &opt_presynaptic_noise, "deviation of noise to add before non-linear transform"),
  OPT_WITH_ARG("-c|--classification-file", opt_set_charp, opt_show_charp,
      &opt_classification_file, "Read class information from this file"),
  OPT_WITH_ARG("-v|--validation-file", opt_set_charp, opt_show_charp,
      &opt_validation_file, "validate using this file"),
  OPT_WITH_ARG("-D|--classification-dir", opt_set_charp, opt_show_charp,
      &opt_classification_dir, "text file paths are relative to here"),
  OPT_WITH_ARG("--learning-style=<n>", opt_set_intval, opt_show_intval,
      &opt_learning_style, "0: weighted, 1: Nesterov, 2: simplified N., "
      "3: classical, 4: adagrad, 5: adadelta, 6: rprop"),
  OPT_WITH_ARG("--ada-ballast", opt_set_floatval, opt_show_floatval,
      &opt_ada_ballast, "adagrad/adadelta accumulators start at this value"),
  OPT_WITH_ARG("--activation", opt_set_intval, opt_show_intval,
      &opt_activation, "1: ReLU, 2: ReSQRT, 5: clipped ReLU"),
  OPT_WITHOUT_ARG("--no-save-net", opt_set_invbool,
      &opt_save_net, "Don't save learnt changes"),
  OPT_WITH_ARG("--epochs=<n>", opt_set_intval, opt_show_intval,
      &opt_epochs, "run for this many epochs"),
  OPT_WITH_ARG("-f|--filename=<file>", opt_set_charp, opt_show_charp,
      &opt_filename, "load/save net here"),

  OPT_ENDTABLE
};

static int
parse_opts(int argc, char *argv[]){
  opt_register_table(options, NULL);
  if (!opt_parse(&argc, argv, opt_log_stderr)){
    exit(1);
  }
  if (argc > 1){
    DEBUG("extraneous arguments:");
    for (int i = 1; i < argc; i++){
      DEBUG("   '%s'", argv[i]);
    }
    opt_usage_and_exit(argv[0]);
  }
  return argc;
}

int
main(int argc, char *argv[]){
  parse_opts(argc, argv);

  u32 flags = (RNN_CHAR_FLAG_CASE_INSENSITIVE | RNN_CHAR_FLAG_UTF8 |
      RNN_CHAR_FLAG_COLLAPSE_SPACE);

  RnnCharClassifiedText *t = new_charmodel_from_filelist(opt_classification_file,
      opt_classification_dir, opt_validation_file, opt_alpha_threshold,
      opt_digit_adjust, opt_alpha_adjust, flags, opt_ignore_start);

  if (opt_verbose >= 1){
    rnn_char_dump_alphabet(t->alphabet);
  }

  RnnCharClassifier *model = malloc(sizeof(RnnCharClassifier));
  model->text = t;
  model->n_training_nets = MAX(opt_multi_tap, 1);
  model->pgm_name = "text-classify";
  model->momentum = opt_momentum;
  model->momentum_soft_start = 2000;
  model->learning_style = opt_learning_style;
  model->images.temporal_pgm_dump = 0;
  model->periodic_weight_noise = 0;
  model->report_interval = 1024;
  model->save_net = opt_save_net;

  struct RnnCharMetadata m;
  m.alphabet = new_utf8_from_codepoints(t->alphabet->points, t->alphabet->len);
  m.collapse_chars = new_utf8_from_codepoints(t->alphabet->collapsed_points,
      t->alphabet->collapsed_len);
  m.utf8 = 1;

  if (opt_filename){
    model->filename = opt_filename;
  }
  else{
    model->filename = rnn_char_construct_net_filename(&m, opt_basename, t->alphabet->len,
        0, opt_hidden_size, t->n_classes);
  }

  u32 net_flags = RNN_NET_FLAG_STANDARD | RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR;
  RecurNN *net = rnn_new(t->alphabet->len, opt_hidden_size, t->n_classes, net_flags,
      opt_rng_seed, opt_logfile, opt_bptt_depth, opt_learn_rate,
      opt_momentum, opt_presynaptic_noise, opt_activation);
  rnn_randomise_weights_auto(net);

  switch(opt_learning_style){
  case RNN_ADAGRAD:
    if (opt_ada_ballast < 0){
      opt_ada_ballast = DEFAULT_ADAGRAD_BALLAST;
    }
    rnn_set_momentum_values(net, opt_ada_ballast);
    break;

  case RNN_ADADELTA:
    if (opt_ada_ballast < 0){
      opt_ada_ballast = DEFAULT_ADADELTA_BALLAST;
    }
    rnn_set_momentum_values(net, opt_ada_ballast);
    break;

  case RNN_RPROP:
    rnn_set_aux_values(net, 1);
    break;
  }

  net->bptt->momentum_weight = 0.5;
  net->metadata = rnn_char_construct_metadata(&m);
  model->net = net;
  model->training_nets = rnn_new_training_set(net, model->n_training_nets);
  /*bar chart column titles */
  DEBUG("n_classes %d", t->n_classes);
  for (int i = 0; i < opt_epochs; i++){
    for (int j = 0; j < t->n_classes; j++){
      for (int k = 0; k < j; k++){
        fputs((k & 1) ? C_GREY : C_WHITE, stderr);
        fputs("\xE2\x94\x82", stderr); /* vertical */
      }
      fputs((j & 1) ? C_GREY : C_WHITE, stderr);
      fputs("\xE2\x95\xad", stderr); /* corner */
      for (int k = j; k < t->n_classes; k++){
        fputs("\xE2\x94\x80", stderr); /* horizontal */
      }
      /* finish with a half horizontal */
      fprintf(stderr, "\xE2\x95\xB4%s" C_NORMAL "\n", t->classes[j]);
    }
    rnn_char_classify_epoch(model);
  }
  if (model->filename && opt_save_net){
    rnn_save_net(model->net, model->filename, 1);
  }
}
