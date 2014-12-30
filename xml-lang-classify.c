/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> GPL2+

This tries to use the RNN to classify the language of text in TEI xml
documents.

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
#include "colour.h"
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <libxml/debugXML.h>
#include "opt-helpers.h"


#define NO_LANG "xx"

#define MAXLEN 20 * 1000 * 1000

static inline void
dump_xmlnode(xmlNode *x){
  xmlDebugDumpOneNode(stderr, x, 1);
}

static inline ALWAYS_INLINE int
lookup_class(char** class_lut, const char *class, int set_if_not_found){
  if (strcmp(class, NO_LANG)){
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

static RnnCharClassBlock *
alloc_langblock_from_xml(xmlNode *el, RnnCharClassBlock *b, const char *lang,
    char **class_lut, const char *parent)
{
  if (el->type != XML_TEXT_NODE){
    char *name = (char *)el->name;
    if (streq(name, "teiHeader")){
      /*teiHeader is full of nonsense; ignore it */
      return b;
    }
    if (streq(parent, "choice") && ! streq(name, "orig")){
      /*<choice> contains alternate versions <orig> and <reg>.*/
      return b;
    }
    if(streq(name, "foreign")){
      /*foreign designations are unreliable*/
      lang = NO_LANG;
    }
    else{
      const char *lang_attr = (char *)xmlGetProp(el, (const xmlChar *)"lang");
      if (lang_attr){
        lang = lang_attr;
      }
    }
  }
  int class_code = lookup_class(class_lut, lang, 1);
  if (class_code != NO_CLASS){
    lang = class_lut[class_code]; /*use the strduped copy, not the xml one*/
  }

  for(xmlNode *c = el->xmlChildrenNode; c; c = c->next){
    if (c->type == XML_TEXT_NODE){
      char *text = (char *)xmlNodeGetContent(c);
      b->class_name = lang;
      b->class_code = class_code;
      b->text = text;
      b->len = strlen(b->text);
      b->next = malloc(sizeof(RnnCharClassBlock));
      b = b->next;
    }
    else {
      b = alloc_langblock_from_xml(c, b, lang, class_lut, (char *)el->name);
    }
  }
  return b;
}

static RnnCharClassBlock *
new_langblocks_from_xml(char *filename, char ***classes, int *n_classes){

  xmlDoc *doc = xmlReadFile(filename, NULL, 0);
  xmlNode *xml = xmlDocGetRootElement(doc);

  RnnCharClassBlock *b = calloc(1, sizeof(RnnCharClassBlock));
  *classes = calloc(256, sizeof(char*));
  (*classes)[NO_CLASS] = NO_LANG;
  RnnCharClassBlock *end = alloc_langblock_from_xml(xml, b, NO_LANG, *classes, "");
  end->next = NULL;
  xmlFreeDoc(doc);
  xmlCleanupParser();

  for (int i = 0; i < 256; i++){
    if (! (*classes)[i]){
      *n_classes = i;
      break;
    }
  }
  return b;
}

static void
free_langblocks(RnnCharClassBlock *b){
  RnnCharClassBlock *next;
  for (; b; b = next){
    next = b->next;
    free((char *)b->text);
    free(b);
    b = next;
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

static RnnCharClassifiedText *
new_charmodel_from_xml(char *filename, double alpha_threshold,
    double digit_adjust, double alpha_adjust, u32 flags)
{
  int n_classes;
  char **classes;
  RnnCharClassBlock *first_block = new_langblocks_from_xml(filename, &classes, &n_classes);
  RnnCharClassifiedChar *classified_text;
  RnnCharAlphabet *alphabet = rnn_char_new_alphabet();
  alphabet->flags = flags;
  int textlen;
  char *fulltext = new_full_text_from_blocks(first_block, &textlen);
  int r = rnn_char_find_alphabet_s(fulltext, textlen, alphabet,
      alpha_threshold, digit_adjust, alpha_adjust);
  free(fulltext);

  if (r){
    DEBUG("could not find alphabet (error %d)", r);
    return NULL;
  }

  classified_text = rnn_char_alloc_classified_text(first_block,
      alphabet, &textlen, 0);

  RnnCharClassifiedText *t = malloc(sizeof(*t));
  t->text = classified_text;
  t->len = textlen;
  t->alphabet = alphabet;
  t->lag = 0;
  t->n_classes = n_classes;
  t->classes = classes;
  free_langblocks(first_block);
  return t;
}


static void
dump_colourised_text(RnnCharClassifiedText *t){
  const char *colour_lut[] = {C_RED, C_GREEN, C_YELLOW, C_BLUE};
  const char *no_class_colour = C_CYAN;
  u8 prev = NO_CLASS;
  printf("%s", no_class_colour);
  u8 n_colours = sizeof(colour_lut) / sizeof(colour_lut[0]);
  RnnCharClassifiedChar *text = t->text;
  char s[5] = {0};
  int *clut = t->alphabet->points;
  int utf8 = t->alphabet->flags & RNN_CHAR_FLAG_UTF8;
  for (int i = 0; i < t->len; i++){
    u8 symbol = text[i].symbol;
    u8 class = text[i].class;
    int code = clut[symbol];
    if (utf8){
      int end = write_utf8_char(code, s);
      s[end] = 0;
    }
    else {
      *s = code;
    }
    if (prev != class){
      prev = class;
      const char *colour = (prev < n_colours) ? colour_lut[class] : no_class_colour;
      printf("%s%s", colour, s);
    }
    else {
      printf("%s", s);
    }
  }
  puts(C_NORMAL);
}

//#define DEFAULT_XML "/home/douglas/corpora/maori-legal-papers/GorLaws.xml"
#define DEFAULT_XML "/home/douglas/corpora/maori-legal-papers/Gov1909Acts.xml"
#define DEFAULT_UTF8 1
#define DEFAULT_COLLAPSE_SPACE 1
#define DEFAULT_CASE_INSENSITIVE 1

static char *opt_xmlfile = DEFAULT_XML;
static bool opt_utf8 = DEFAULT_UTF8;
static bool opt_collapse_space = DEFAULT_COLLAPSE_SPACE;
static bool opt_case_insensitive = DEFAULT_CASE_INSENSITIVE;
static double opt_alpha_threshold = 1e-4;
static double opt_alpha_adjust = 3.0;
static double opt_digit_adjust = 1.0;
static int opt_lag = 0;
static bool opt_dump_colour = false;
static int opt_verbose = 0;
static uint opt_multi_tap = 20;
static uint opt_hidden_size = 199;
static u64 opt_rng_seed = 11;
static char * opt_logfile = NULL;
static uint opt_bptt_depth = 40;
static float opt_learn_rate = 0.001;
static float opt_momentum = 0.93;
static char * opt_basename = "xml-lang-classify";
static float opt_presynaptic_noise = 0;


static struct opt_table options[] = {
  OPT_WITHOUT_ARG("-h|--help", opt_usage_and_exit,
      ": Rnn classification of text at the character level",
      "Print this message."),

  OPT_WITH_ARG("-H|--hidden-size=<n>", opt_set_uintval, opt_show_uintval,
      &opt_hidden_size, "number of hidden nodes"),
  OPT_WITH_ARG("-r|--rng-seed=<seed>", opt_set_ulongval_bi, opt_show_ulongval_bi,
      &opt_rng_seed, "RNG seed (-1 for auto)"),
  OPT_WITH_ARG("-x|--xmlfile=<file>", opt_set_charp, opt_show_charp, &opt_xmlfile,
      "operate on this XML file"),
  OPT_WITHOUT_ARG("--case-sensitive", opt_set_invbool,
      &opt_case_insensitive, "Treat capitals as their separate symbols"),
  OPT_WITHOUT_ARG("--case-insensitive", opt_set_bool,
      &opt_case_insensitive, "Treat capitals as lower case characters (ASCII only)"),
  OPT_WITHOUT_ARG("--utf8", opt_set_bool,
      &opt_utf8, "Parse text as UTF8"),
  OPT_WITHOUT_ARG("--no-utf8", opt_set_invbool,
      &opt_utf8, "Parse text as 8 bit symbols"),
  OPT_WITHOUT_ARG("--no-collapse-space", opt_set_invbool,
      &opt_collapse_space, "Predict whitespace characters individually"),
  OPT_WITHOUT_ARG("--collapse-space", opt_set_bool,
      &opt_collapse_space, "Runs of whitespace collapse to single space"),
  OPT_WITH_ARG("--find-alphabet-threshold", opt_set_doubleval, opt_show_doubleval,
      &opt_alpha_threshold, "minimum frequency for character to be included"),
  OPT_WITH_ARG("--find-alphabet-digit-adjust", opt_set_doubleval, opt_show_doubleval,
      &opt_digit_adjust, "adjust digit frequency for alphabet calculations"),
  OPT_WITH_ARG("--find-alphabet-alpha-adjust", opt_set_doubleval, opt_show_doubleval,
      &opt_alpha_adjust, "adjust letter frequency for alphabet calculation"),
  OPT_WITH_ARG("-L|--lag", opt_set_intval, opt_show_intval,
      &opt_lag, "classify character this far back"),
  OPT_WITHOUT_ARG("--dump-colour", opt_set_bool,
      &opt_dump_colour, "Print text in colour showing training classes"),
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

  OPT_ENDTABLE
};

static void
parse_opts(int argc, char *argv[]){
  opt_register_table(options, NULL);
  if (!opt_parse(&argc, argv, opt_log_stderr)){
    exit(1);
  }
  if (argc > 1){
    DEBUG("unused arguments:");
    for (int i = 1; i < argc; i++){
      DEBUG("   '%s'", argv[i]);
    }
    opt_usage_and_exit(NULL);
  }
}


int
main(int argc, char *argv[]){
  parse_opts(argc, argv);

  u32 flags = (
      (opt_case_insensitive ? RNN_CHAR_FLAG_CASE_INSENSITIVE : 0) |
      (opt_utf8 ? RNN_CHAR_FLAG_UTF8 : 0) |
      (opt_collapse_space ? RNN_CHAR_FLAG_COLLAPSE_SPACE : 0));

  RnnCharClassifiedText *t = new_charmodel_from_xml(opt_xmlfile,
      opt_alpha_threshold, opt_digit_adjust, opt_alpha_adjust, flags);

  if (opt_lag){
    rnn_char_adjust_text_lag(t, opt_lag);
  }
  if (opt_dump_colour){
    dump_colourised_text(t);
  }
  if (opt_verbose >= 1){
    rnn_char_dump_alphabet(t->alphabet);
  }

  RnnCharClassifier *model = malloc(sizeof(RnnCharModel));
  model->text = t;
  model->n_training_nets = MAX(opt_multi_tap, 1);
  model->pgm_name = "xml-lang-classify";
  model->momentum = opt_momentum;
  model->momentum_soft_start = 2000;
  model->learning_style = 0;
  model->images.temporal_pgm_dump = 0;
  model->periodic_weight_noise = 0;
  model->report_interval = 1024;
  model->save_net = 0;

  struct RnnCharMetadata m;
  if (opt_utf8){
    m.alphabet = new_utf8_from_codepoints(t->alphabet->points, t->alphabet->len);
    m.collapse_chars = new_utf8_from_codepoints(t->alphabet->collapsed_points,
        t->alphabet->collapsed_len);
    m.utf8 = 1;
  }
  else {
    m.alphabet = new_bytes_from_codepoints(t->alphabet->points, t->alphabet->len);
    m.collapse_chars = new_bytes_from_codepoints(t->alphabet->collapsed_points,
        t->alphabet->collapsed_len);
    m.utf8 = 0;
  }

  model->filename = rnn_char_construct_net_filename(&m, opt_basename, t->alphabet->len,
      0, opt_hidden_size, t->n_classes);

  u32 net_flags = RNN_NET_FLAG_STANDARD | RNN_NET_FLAG_BPTT_ADAPTIVE_MIN_ERROR;
  RecurNN *net = rnn_new(t->alphabet->len, opt_hidden_size, t->n_classes, net_flags,
      opt_rng_seed, opt_logfile, opt_bptt_depth, opt_learn_rate,
      opt_momentum, opt_presynaptic_noise, RNN_RELU);
  rnn_randomise_weights_auto(net);

  net->bptt->momentum_weight = 0.5;
  net->metadata = rnn_char_construct_metadata(&m);
  model->net = net;
  model->training_nets = rnn_new_training_set(net, model->n_training_nets);

  for (;;){
    rnn_char_classify_epoch(model);
  }
}
