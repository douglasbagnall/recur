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
  dump_xmlnode(el);
  if (el->type == XML_ELEMENT_NODE && streq((char *)el->name, "foreign")){
    lang = NO_LANG;
  }
  else{
    const char *lang_attr = (char *)xmlGetProp(el, (const xmlChar *)"lang");
    if (lang_attr){
      lang = lang_attr;
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
      //DEBUG("found text. %s lang %s len %d b %p next %p", text, lang, b->len, b, b->next);
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
    /*DEBUG("cumlen %d b->len %d size %d b %p b->next %p",
        cumlen, b->len, size, b, b->next);
        DEBUG("text %s", b->text);*/
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
  printf("%s", text);
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
  DEBUG("fulltext is %d characters. first block is %p, %d long", textlen,
      first_block, first_block->len);

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

//#define XMLNAME "/home/douglas/corpora/maori-legal-papers/GorLaws.xml"
#define XMLNAME "/home/douglas/corpora/maori-legal-papers/Gov1909Acts.xml"

int
main(int argc, char *argv[]){
  u32 flags = (RNN_CHAR_FLAG_CASE_INSENSITIVE |
      RNN_CHAR_FLAG_UTF8 |
      RNN_CHAR_FLAG_COLLAPSE_SPACE);

  char *filename = XMLNAME;
  RnnCharClassifiedText *t = new_charmodel_from_xml(filename, 1e-4,
      0.5, 3.0, flags);
  dump_colourised_text(t);

  rnn_char_dump_alphabet(t->alphabet, t->a_len, flags & RNN_CHAR_FLAG_UTF8);
  rnn_char_dump_alphabet(t->collapse_chars, t->c_len, flags & RNN_CHAR_FLAG_UTF8);
}
