/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> GPL2+

This uses the RNN to predict the next character in a text sequence.

Unlike most of the Recur repository, this file is licensed under the GNU
General Public License, version 2 or greater. That is because it is linked to
ccan/opt which is also GPL2+.

Because of ccan/opt, --help will tell you something.
*/

#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include "ccan/opt/opt.h"
#include "charmodel.h"
#include "opt-helpers.h"

static char * opt_filename = NULL;
static float opt_bias = 0;
static uint opt_chars = 72;

static struct opt_table options[] = {
  OPT_WITH_ARG("-f|--filename=<file>", opt_set_charp, opt_show_charp, &opt_filename,
      "load net from this file"),
  OPT_WITH_ARG("-B|--bias", opt_set_floatval, opt_show_floatval,
      &opt_bias, "bias toward probable characters "
      "(100 == deterministic)"),
  OPT_WITH_ARG("-n|--length=<chars>", opt_set_uintval, opt_show_uintval,
      &opt_chars, "confabulate this many characters"),

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

  /* I am not soure if this is necessary */
  init_rand64_maybe_randomly(&net->rng, -1);

  /*XXX this could be done in small chunks */
  int byte_len = opt_chars * 4 + 5;
  char *t = malloc(byte_len);

  rnn_char_confabulate(net, t, opt_chars, byte_len,
      alphabet, opt_bias);
  fputs(t, stdout);
  fputs("\n", stdout);
  free(t);
}