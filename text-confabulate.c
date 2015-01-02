/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> GPL2+

This generates text using an RNN trained by text-predict.

Unlike most of the Recur repository, this file is licensed under the GNU
General Public License, version 2 or greater. That is because it is linked to
ccan/opt which is also GPL2+.

Because of ccan/opt, --help will tell you something.
*/

#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include "charmodel.h"
#include "colour.h"
#include "opt-helpers.h"

static char * opt_filename = NULL;
static float opt_bias = 0;
static uint opt_chars = 72;
static char *opt_prefix = NULL;
static bool opt_show_prefix = false;
static char *opt_until = NULL;
static char *opt_wait_for = NULL;
static s64 opt_rng_seed = -1;

static struct opt_table options[] = {
  OPT_WITH_ARG("-f|--filename=<file>", opt_set_charp, opt_show_charp, &opt_filename,
      "load net from this file"),
  OPT_WITH_ARG("-B|--bias", opt_set_floatval, opt_show_floatval,
      &opt_bias, "bias toward probable characters "
      "(100 == deterministic)"),
  OPT_WITH_ARG("-n|--length=<chars>", opt_set_uintval, opt_show_uintval,
      &opt_chars, "confabulate this many characters"),
  OPT_WITH_ARG("-p|--prefix=<chars>", opt_set_charp, opt_show_charp, &opt_prefix,
      "seed the confabulator with this"),
  OPT_WITHOUT_ARG("--show-prefix", opt_set_bool, &opt_show_prefix,
      "print the prefix (if any)"),
  OPT_WITH_ARG("-u|--until=<char>", opt_set_charp, opt_show_charp, &opt_until,
      "stop when this charactor appears"),
  OPT_WITH_ARG("--wait-for=<char>", opt_set_charp, opt_show_charp, &opt_wait_for,
      "don't start until this charactor appears"),
  OPT_WITH_ARG("-r|--rng-seed=<seed>", opt_set_longval_bi, opt_show_longval_bi,
      &opt_rng_seed, "RNG seed (default: -1 for auto)"),

  OPT_WITHOUT_ARG("-h|--help", opt_usage_and_exit,
      ": Confabulate text using previously trained RNN",
      "Print this message."),
  OPT_ENDTABLE
};


int
main(int argc, char *argv[]){
  opt_register_table(options, NULL);
  if (!opt_parse(&argc, argv, opt_log_stderr)){
    exit(1);
  }
  if (argc > 1){
    DEBUG("unused arguments:");
    for (int i = 1; i < argc; i++){
      DEBUG("   '%s'", argv[i]);
    }
    opt_usage(argv[0], NULL);
  }
  RecurNN *net = rnn_load_net(opt_filename);
  RnnCharAlphabet *alphabet = rnn_char_new_alphabet_from_net(net);

  init_rand64_maybe_randomly(&net->rng, opt_rng_seed);

  int prev_char = 0;

  if (opt_prefix){
    u8 *prefix_text;
    int prefix_len;
    prefix_text = (u8*)strdup(opt_prefix);
    int raw_len = strlen(opt_prefix);
    rnn_char_collapse_buffer(alphabet, prefix_text,
        raw_len, &prefix_len);
    prev_char = rnn_char_prime(net, alphabet, prefix_text, prefix_len);
    if (opt_show_prefix){
      printf(C_CYAN "%s" C_NORMAL, opt_prefix);
    }
  }

  /*XXX this could be done in small chunks */
  int byte_len = opt_chars * 4 + 5;
  char *t = malloc(byte_len);

  int stop_point = -1;
  int start_point = -1;
  if (opt_until){
    stop_point = rnn_char_get_codepoint(alphabet, opt_until);
  }
  if (opt_wait_for){
    start_point = rnn_char_get_codepoint(alphabet, opt_wait_for);
  }

  rnn_char_confabulate(net, t, opt_chars, byte_len,
      alphabet, opt_bias, &prev_char, start_point, stop_point);
  fputs(t, stdout);
  fputs("\n", stdout);
  free(t);
}
