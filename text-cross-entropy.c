/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> GPL2+

Print the cross entropy of the text in the given files against the predictions
of a net trained by text-predict.

This file is GPL2+, due to the ccan/opt link.
*/

#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include "opt-helpers.h"
#include "charmodel.h"
#include "utf8.h"
#include "charmodel-helpers.h"
#include "colour-spectrum.h"

#include "colour.h"

static char *opt_filename = NULL;
static int opt_min_length = 0;
static int opt_ignore_first = 0;
static char *opt_prefix = NULL;
static float opt_colour_scale = 0;
static float opt_colour_decay = 1;
static bool opt_colour_24_bit = false;

static struct opt_table options[] = {
  OPT_WITH_ARG("-f|--filename=<file>", opt_set_charp, opt_show_charp, &opt_filename,
      "load net from this file"),
  OPT_WITH_ARG("-m|--min-length=<n>", opt_set_intval, opt_show_intval, &opt_min_length,
      "ignore texts shorter than this"),
  OPT_WITH_ARG("-i|--ignore-first=<n>", opt_set_intval, opt_show_intval, &opt_ignore_first,
      "ignore first n characters"),
  OPT_WITH_ARG("-p|--prefix=<chars>", opt_set_charp, opt_show_charp, &opt_prefix,
      "pretend each file starts with this"),
  OPT_WITH_ARG("-c|--colour-scale", opt_set_floatval, opt_show_floatval,
      &opt_colour_scale, "colourise text showing cross entropy"),
  OPT_WITH_ARG("-d|--colour-decay", opt_set_floatval01, opt_show_floatval,
      &opt_colour_decay, "set < 1 for exponential decay in colour"),
  OPT_WITHOUT_ARG("--colour-24-bit", opt_set_bool,
      &opt_colour_24_bit, "use a 24 bit RGB colour spectrum"),

  OPT_WITHOUT_ARG("-h|--help", opt_usage_and_exit,
      "-f NET TEXTFILE [TEXT...] \n"
      "Print the cross-entropy of each text file",
      "Print this message."),
  OPT_ENDTABLE
};


static double
colourise_text(RecurNN *net, RnnCharAlphabet *alphabet, u8 *text, int len,
    int skip, u8 *prefix_text, int prefix_len, float scale, float decay,
    bool use_24_bit){
  double entropy = 0;
  float error[net->output_size];
  int i;
  int n_chars = net->output_size;
  char buffer[6];
  bool utf8 = alphabet->flags & RNN_CHAR_FLAG_UTF8;

  const char *normal_colour = C_NORMAL BG_NORMAL;
  const char **colours;
  int n_colours;
  if (use_24_bit) {
    colours = COLOURS_24;
    n_colours = N_COLOURS_24;
  }
  else {
    colours = COLOURS_256;
    n_colours = N_COLOURS_256;
  }

  if (prefix_text){
    rnn_char_prime(net, alphabet, prefix_text, prefix_len);
  }
  puts(normal_colour);
  for (i = 0; i < skip; i++){
    one_hot_opinion(net, text[i], 0);
    write_possibly_utf8_char(alphabet->points[text[i]], buffer, utf8);
  }
  write_possibly_utf8_char(alphabet->points[text[skip]], buffer, utf8);
  float rolling_log_p = 1;
  for (; i < len - 1; i++){
    float *answer = one_hot_opinion(net, text[i], 0);
    softmax(error, answer, n_chars);
    int next = text[i + 1];
    float e = error[next];
    int j = write_possibly_utf8_char(alphabet->points[next], buffer, utf8);

    float log_p = -capped_log2f(e);
    rolling_log_p = rolling_log_p * (1.0f - decay) + log_p * decay;
    uint colour_index = MIN(rolling_log_p * scale, n_colours - 1);
    //printf("(%.3f %d)", log_p, colour_index);
    fputs(colours[colour_index], stdout);
    if (colour_index > 20){
      fputs(ITALIC, stdout);
      fwrite(buffer, 1, j, stdout);
      fputs(STANDARD, stdout);
    }
    else {
      fwrite(buffer, 1, j, stdout);
    }
    entropy += log_p;
  }
  entropy /= (len - skip - 1);
  puts(normal_colour);
  return entropy;
}


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
      double entropy;
      if (opt_colour_scale){
        entropy = colourise_text(net, alphabet, text, len, opt_ignore_first,
            prefix_text, prefix_len, opt_colour_scale, opt_colour_decay,
            opt_colour_24_bit);
      }
      else {
        entropy = rnn_char_cross_entropy(net, alphabet, text, len, opt_ignore_first,
            prefix_text, prefix_len);
      }
      fprintf(stdout, "%s %.5f\n", filename, entropy);
      count++;
    }
    free(text);
  }
  DEBUG("processed %d texts", count);
  return 0;
}
