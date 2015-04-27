/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> GPL2+

Print the cross entropy of the text in the given files against the predictions
of a net trained by text-predict.

This file is GPL2+, due to the ccan/opt link.
*/

#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include "opt-helpers.h"
#include "charmodel.h"

static char *opt_filename = NULL;
static int opt_min_length = 0;
static int opt_ignore_first = 0;
static char *opt_prefix = NULL;

static struct opt_table options[] = {
  OPT_WITH_ARG("-f|--filename=<file>", opt_set_charp, opt_show_charp, &opt_filename,
      "load net from this file"),
  OPT_WITH_ARG("-m|--min-length=<n>", opt_set_intval, opt_show_intval, &opt_min_length,
      "ignore texts shorter than this"),
  OPT_WITH_ARG("-i|--ignore-first=<n>", opt_set_intval, opt_show_intval, &opt_ignore_first,
      "ignore first n characters"),
  OPT_WITH_ARG("-p|--prefix=<chars>", opt_set_charp, opt_show_charp, &opt_prefix,
      "pretend each file starts with this"),

  OPT_WITHOUT_ARG("-h|--help", opt_usage_and_exit,
      "-f NET TEXTFILE [TEXT...] \n"
      "Print the cross-entropy of each text file",
      "Print this message."),
  OPT_ENDTABLE
};

int
main(int argc, char *argv[]){
  opt_register_table(options, NULL);
  if (!opt_parse(&argc, argv, opt_log_stderr)){
    exit(1);
  }
  if (argc == 1){
    DEBUG("No text files to evaluate!");
    opt_usage_and_exit(argv[0]);
  }
  DEBUG("given %d arguments", argc - 1);

  RecurNN *net = rnn_load_net(opt_filename);
  RnnCharAlphabet *alphabet = rnn_char_new_alphabet_from_net(net);
  init_rand64_maybe_randomly(&net->rng, -1);

  u8 *prefix_text;
  int prefix_len;

  if (opt_prefix){
    prefix_text = (u8*)strdup(opt_prefix);
    int raw_len = strlen(opt_prefix);
    rnn_char_collapse_buffer(alphabet, prefix_text,
        raw_len, &prefix_len, NULL);
  }
  else {
    prefix_text = NULL;
    prefix_len = 0;
  }

  int len;
  int count = 0;

  for (int i = 1; i < argc; i++){
    const char *filename = argv[i];
    u8* text = rnn_char_alloc_collapsed_text(filename, alphabet, &len, 3);
    if (len >= opt_min_length){
      double entropy = rnn_char_cross_entropy(net, alphabet, text, len, opt_ignore_first,
          prefix_text, prefix_len);
      fprintf(stdout, "%s %.5f\n", filename, entropy);
      count++;
    }
    free(text);
  }
  DEBUG("processed %d texts", count);
  return 0;
}
