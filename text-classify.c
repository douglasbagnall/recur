/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> GPL2+

This uses the RNN to predict the next character in a text sequence.

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
#include "ccan/opt/opt.h"
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
    char **class_lut)
{
  if (el->type != XML_TEXT_NODE){
    char *name = (char *)el->name;
    if (streq(name, "teiHeader")){
      /*teiHeader is full of nonsense; ignore it */
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
      b = alloc_langblock_from_xml(c, b, lang, class_lut);
    }
  }
  return b;
}

static RnnCharClassBlock *
new_langblocks_from_xml(char *filename){

  xmlDoc *doc = xmlReadFile(filename, NULL, 0);
  xmlNode *xml = xmlDocGetRootElement(doc);

  RnnCharClassBlock *b = calloc(1, sizeof(RnnCharClassBlock));
  char *class_lut[257] = {0};
  class_lut[NO_CLASS] = NO_LANG;
  RnnCharClassBlock *end = alloc_langblock_from_xml(xml, b, NO_LANG, class_lut);
  end->next = NULL;
  xmlFreeDoc(doc);
  xmlCleanupParser();
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
  RnnCharClassBlock *first_block = new_langblocks_from_xml(filename);
  RnnCharClassifiedChar *classified_text;
  int *alphabet = calloc(257, sizeof(int));
  int *collapse_chars = calloc(257, sizeof(int));
  int a_len;
  int c_len;
  int textlen;
  char *fulltext = new_full_text_from_blocks(first_block, &textlen);
  int r = rnn_char_find_alphabet_s(fulltext, textlen, alphabet, &a_len,
      collapse_chars, &c_len, alpha_threshold, digit_adjust,
      alpha_adjust, flags);
  free(fulltext);

  if (r){
    DEBUG("could not find alphabet (error %d)", r);
    return NULL;
  }

  classified_text = rnn_char_alloc_classified_text(first_block,
      alphabet, a_len, collapse_chars, c_len,
      &textlen, flags);

  RnnCharClassifiedText *t = malloc(sizeof(*t));
  t->text = classified_text;
  t->len = textlen;
  t->alphabet = alphabet;
  t->a_len = a_len;
  t->collapse_chars = collapse_chars;
  t->c_len = c_len;
  t->flags = flags;
  t->lag = 0;
  free_langblocks(first_block);
  return t;
}


static void
dump_colourised_text(RnnCharClassifiedText *t){
  const char *colour_lut[] = {C_RED, C_GREEN, C_YELLOW, C_BLUE};
  const char *no_class_colour = C_CYAN;
  u8 prev = NO_CLASS;
  u8 n_colours = sizeof(colour_lut) / sizeof(colour_lut[0]);
  RnnCharClassifiedChar *text = t->text;
  char s[5] = {0};
  int *clut = t->alphabet;
  int utf8 = t->flags & RNN_CHAR_FLAG_UTF8;
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

static struct opt_table options[] = {
  OPT_WITHOUT_ARG("-h|--help", opt_usage_and_exit,
      ": Rnn classification of text at the character level",
      "Print this message."),

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
  OPT_WITH_ARG("-l|--lag", opt_set_intval, opt_show_intval,
      &opt_lag, "classify character this far back"),
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
  rnn_char_dump_alphabet(t->alphabet, t->a_len, flags & RNN_CHAR_FLAG_UTF8);
  rnn_char_dump_alphabet(t->collapse_chars, t->c_len, flags & RNN_CHAR_FLAG_UTF8);
}
