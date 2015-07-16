/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> GPL2+

Emit class probabilities for a set of files, based on a net trained by
text-classify.

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
#include "charmodel-helpers.h"
#include "utf8.h"
#include "colour.h"
#include "opt-helpers.h"

#define UNCLASSIFIED "*unclassified*"

static int opt_ignore_start = 0;
static int opt_min_length = 0;
static char *opt_filename = NULL;
static char *opt_test_dir = NULL;


static struct opt_table options[] = {
  OPT_WITH_ARG("-m|--min-length=<n>", opt_set_intval, opt_show_intval, &opt_min_length,
      "ignore texts shorter than this"),
  OPT_WITH_ARG("-i|--ignore-start", opt_set_intval, opt_show_intval,
      &opt_ignore_start, "don't classify this many first characters per block"),
  OPT_WITH_ARG("-f|--filename=<file>", opt_set_charp, opt_show_charp,
      &opt_filename, "load/save net here"),
  OPT_WITH_ARG("-t|--test-dir=<dir>", opt_set_charp, opt_show_charp,
      &opt_test_dir, "emit scores for files in this directory"),

  OPT_WITHOUT_ARG("-h|--help", opt_usage_and_exit,
      "-f NET TEXTFILE [TEXTFILE...] \n"
      "Print classification probabilities of documents",
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

  int len = 0;
  int count = 0;

  if (opt_min_length <= opt_ignore_start){
    DEBUG("hey! --min-length=%d <= --ignore-start=%d! Fixing.. now its %d.",
        opt_min_length, opt_ignore_start, opt_ignore_start + 1);
    opt_min_length = opt_ignore_start + 1;
  }
  float sum[net->output_size];
  float sumsq[net->output_size];
  float mean[net->output_size];
  float stddev[net->output_size];

  for (int i = 1; i < argc; i++){
    const char *filename = argv[i];
    u8* text = rnn_char_load_new_encoded_text(filename, alphabet, &len, 3);


    if (len >= opt_min_length){
      memset(sum, 0, net->output_size * sizeof(float));
      memset(sumsq, 0, net->output_size * sizeof(float));
      int j, k;
      for (j = 0; j < opt_ignore_start; j++){
        one_hot_opinion(net, text[j], 0);
      }
      for (j = opt_ignore_start; j < len; j++){
        float *raw = one_hot_opinion(net, text[j], 0);
        float *answer = mean;
        softmax(answer, raw, net->output_size);
        for (k = 0; k < net->output_size; k++){
          float a = answer[k];
          sum[k] += a;
          sumsq[k] += a * a;
        }
      }
      for (k = 0; k < net->output_size; k++){
        float m = sum[k] / (len - opt_ignore_start);
        stddev[k] = sqrtf(sumsq[k] / (len - opt_ignore_start) - m * m);
        mean[k] = m;
      }

      printf("%s mean: ", filename);
      for (k = 0; k < net->output_size; k++){
        printf("%.3e ", mean[k]);
      }
      printf(" stddev: ");
      for (k = 0; k < net->output_size; k++){
        printf("%.3e ", stddev[k]);
      }
      puts("\n");
    }
    free(text);
  }
  DEBUG("processed %d texts", count);
  return 0;
}
