/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL
-*- coding: utf-8 -*-

Tests some of the functions in ../charmodel.[ch]
*/
#include "path.h"
#include "../charmodel.h"
#include "../utf8.h"

#define C_NORMAL  "\033[00m"
#define C_DARK_RED  "\033[00;31m"
#define C_RED "\033[01;31m"
#define C_DARK_GREEN  "\033[00;32m"
#define C_GREEN  "\033[01;32m"
#define C_YELLOW  "\033[01;33m"
#define C_DARK_YELLOW  "\033[00;33m"
#define C_DARK_BLUE  "\033[00;34m"
#define C_BLUE  "\033[01;34m"
#define C_PURPLE  "\033[00;35m"
#define C_MAGENTA  "\033[01;35m"
#define C_DARK_CYAN  "\033[00;36m"
#define C_CYAN  "\033[01;36m"
#define C_GREY  "\033[00;37m"
#define C_WHITE  "\033[01;37m"

#define C_REV_RED "\033[01;41m"

#define EREWHON_TEXT TEST_DATA_DIR "/erewhon.txt"
#define LGPL_TEXT TEST_DATA_DIR "/../licenses/LGPL-2.1"
#define WAI1874_TEXT TEST_DATA_DIR "/Wai1874NgaM.txt"

#define PUT(format, ...) fprintf(stderr, (format),## __VA_ARGS__)

typedef struct {
  double threshold;
  const char *alphabet;
  const char *collapse;
  const char first_char;
  char *whitespace;
  char *filename;
  int ignore_case;
  int utf8;
  int collapse_space;
} ab_test;

//Erewhon chars: note space at start:
// etaonihsrdlucmwfygpb,v.k-;x"qj'?:z)(_1!0*872&{}695/34[]@
static const ab_test ab_test_cases[] = {
  {
    .threshold = 3e-4,
    .alphabet = "z etaonihsrdlucmwfygpb,v.k-;x\"qj'?:",
    .collapse = ")(_1!0*872&{}695/34[]@",
    .first_char = 'z',
    .whitespace = " ",
    .filename = EREWHON_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },
  {
    .threshold = 1e-4,
    .alphabet = "1etaonihsrdlucmwfygpb,v.k-;x\"qj'?:z)(_ ",
    .collapse = "!0*872&{}695/34[]@",
    .first_char = '1',
    .whitespace = " ",
    .filename = EREWHON_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },
  {
    .threshold = 3e-5,
    .alphabet = " etaonihsrdlucmwfygpb,v.k-;x\"qj'?:z)(_1!0*872&{",
    .collapse = "}695/34[]@",
    .first_char = '{',
    .whitespace = " ",
    .filename = EREWHON_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },
  { /*high threshold*/
    .threshold = 0.1,
    .alphabet = "t e",
    .collapse = "aonihsrdlucmwfygpb,v.k-;x\"qj'?:z)(_1!0*872&}{695/34][@",
    .first_char = 't',
    .whitespace = " ",
    .filename = EREWHON_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },
  { /*low threshold -> no collapsed characters */
    .threshold = 1e-7,
    .alphabet = " !\"&'()*,-./0123456789:;?@[]_abcdefghijklmnopqrstuvwxyz{}",
    .collapse = "",
    .first_char = 0,
    .whitespace = " ",
    .filename = EREWHON_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },

  /*case insensitive */
  {
    .threshold = 1e-4,
    .alphabet = "1 etaonhisrdlucmwfygpb,v.Ik-;Tx\"EAqjH'MSWN?C:BOP()zRFY_LDG",
    .collapse = "!UX0*VQ87ZK2J&}{695/34][@",
    .first_char = '1',
    .whitespace = " ",
    .filename = EREWHON_TEXT,
    .ignore_case = 0,
    .utf8 = 0,
    .collapse_space = 1
  },

  /*un-collapsed space */
  {
    .threshold = 1e-4,
    .alphabet = "1etaonihsrdlucmwfygpb,v.k-;x\"qj'?:z)(_ \n\r",
    .collapse = "!0*872&{}695/34[]@",
    .first_char = '1',
    .whitespace = " ",
    .filename = EREWHON_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 0
  },

  /*utf-8 treatment of pure ASCII text -- should just work the same */
  {
    .threshold = 1e-4,
    .alphabet = "1etaonihsrdlucmwfygpb,v.k-;x\"qj'?:z)(_ ",
    .collapse = "!0*872&{}695/34[]@",
    .first_char = '1',
    .whitespace = " ",
    .filename = EREWHON_TEXT,
    .ignore_case = 1,
    .utf8 = 1,
    .collapse_space = 1
  },

  {/*LGPL text, also ascii only */
    .threshold = 1e-4,
    .alphabet = "4 etiorasnhlcduyfbpmwg,v.k)\"x1(q;2j-/'0:96><35",
    .collapse = "87![]z`",
    .first_char = '4',
    .whitespace = " ",
    .filename = LGPL_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },

  {/*Wai1874 text, UTF-8 */
    .threshold = 1e-4,
    .alphabet = "' aiteokhrnu.mgpw<>,1-0£sd42₤367859:)(;ā—v\"c&bjē*/l",
    .collapse = "…yxīōü",
    .first_char = '\'',
    .whitespace = " ",
    .filename = WAI1874_TEXT,
    .ignore_case = 1,
    .utf8 = 1,
    .collapse_space = 1
  },

  {/*Wai1874 text, ignore case */
    .threshold = 1e-4,
    .alphabet = "' aietoknrh.ugmp<>Kw,1MTH-W0RPN£sd42A₤36I785OE9:)(;ā—\"vUVcB&JlS*/ē",
    .collapse = "yD…xüXōCGī",
    .first_char = '\'',
    .whitespace = " ",
    .filename = WAI1874_TEXT,
    .ignore_case = 0,
    .utf8 = 1,
    .collapse_space = 1
  },

  {/*Wai1874 utf-8 text,  preserve whitespace */
    .threshold = 1e-4,
    .alphabet = "'\n\r \"&()*,-./0123456789:;<>abcdeghijklmnoprstuvw£āē—₤",
    .collapse = "xyüīō…",
    .first_char = '\'',
    .whitespace = " ",
    .filename = WAI1874_TEXT,
    .ignore_case = 1,
    .utf8 = 1,
    .collapse_space = 0
  },

  {/*utf-8 parsed as bytes */
    .threshold = 1e-4,
    .alphabet = ("' aiteokhrnu.mgpw><,1-0\xa3\xc2s\xe2""d42\xa4"
        "36785:9\xc4();\x80\x81\x82v\x94\"c&bj*/\x93l"),
    .collapse = "\xa6\x8dxy\xc3\xbc\xc5\xab",
    .first_char = '\'',
    .whitespace = " ",
    .filename = WAI1874_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },
  {/*utf-8 parsed as bytes, preserving whitespace (testing \n\r) */
    .threshold = 1e-4,
    .alphabet = ("' aieto\n\rknrh.ugmp><Kw,1MTH-W0RPN\xc2\xa3\xe2sd42A\x82\xa4"
        "36I785OE:9\xc4();\x80\x81\x94\"vVUc&BJ*/\x93lS"),
    .collapse = "\xa6yD\xc5\x8d\xc3x\xbcXC\xabG",
    .first_char = '\'',
    .whitespace = " ",
    .filename = WAI1874_TEXT,
    .ignore_case = 0,
    .utf8 = 0,
    .collapse_space = 0
  },

  {
    .filename = NULL
  }
};

static inline int
print_code_list_diff(const int *a, int a_len, const int *b, int b_len, int utf8)
{
  int i, j;
  int b2[b_len];
  char s[8];
  memcpy(b2, b, b_len * sizeof(int));
  int diff = 0;
  PUT(C_NORMAL "-->");
  for (i = 0; i < a_len; i++){
    int x = a[i];
    int found = 0;
    for (j = 0; j < b_len; j++){
      if (b2[j] == x){
        b2[j] = 0;
        found = 1;
        break;
      }
    }
    int end;
    if (utf8){
      end = write_utf8_char(x, s);
    }
    else {
      end = write_escaped_char(x, s);
    }
    s[end] = 0;
    if (found){
      PUT(C_GREEN "%s", s);
    }
    else {
      PUT(C_RED "%s", s);
      diff++;
    }
  }
  PUT(C_MAGENTA);
  for (j = 0; j < b_len; j++){
    int x = b2[j];
    if (x){
      int end = utf8 ? write_utf8_char(x, s) : write_escaped_char(x, s);
      s[end] = 0;
      PUT("%s", s);
      diff++;
    }
  }
  PUT(C_NORMAL "<--diff is %d\n", diff);
  return diff;
}



static inline void
dump_alphabet(int *alphabet, int len, int utf8){
  char *s;
  if (utf8){
    s = new_utf8_from_codepoints(alphabet, len);
  }
  else{
    s = new_bytes_from_codepoints(alphabet, len);
  }
  DEBUG(C_DARK_YELLOW "»»" C_NORMAL "%s" C_DARK_YELLOW "««" C_NORMAL, s);
  free(s);
  for (int i = 0; i < len; i++){
    PUT("%d, ", alphabet[i]);
  }
  PUT("\n");
}

static int
test_alphabet_finding(void){
  int i;
  int errors = 0;
  int alphabet[257];
  int collapse_chars[257];
  int target_alphabet[257];
  int target_collapse_chars[257];
  int a_len, c_len;
  int ta_len, tc_len;
  for (i = 0; ; i++){
    const ab_test *a = &ab_test_cases[i];
    if (! a->filename){
      break;
    }
    int err = rnn_char_find_alphabet(a->filename, alphabet, &a_len,
        collapse_chars, &c_len, a->threshold, a->ignore_case,
        a->collapse_space, a->utf8);
    if (err){
      errors++;
      continue;
    }
    DEBUG(C_CYAN "%s" C_NORMAL " threshold %f, case %s, %s, %s space",
        a->filename, a->threshold,
        a->ignore_case ? "insensitive" : "sensitive",
        a->utf8 ? "utf8" : "bytes",
        a->collapse_space ? "collapsed" : "preserved");
    if (a->utf8){
      ta_len = fill_codepoints_from_utf8(target_alphabet, 256, a->alphabet);
      tc_len = fill_codepoints_from_utf8(target_collapse_chars, 256, a->collapse);
    }
    else {
      ta_len = fill_codepoints_from_bytes(target_alphabet, 256, a->alphabet);
      tc_len  = fill_codepoints_from_bytes(target_collapse_chars, 256,
          a->collapse);
    }
    int e = 0;
    DEBUG(C_GREEN "green" C_NORMAL ": in alphabet and target. "
        C_RED "red" C_NORMAL ": in found alphabet only. "
        C_MAGENTA "magenta" C_NORMAL ": in target only.");

    PUT(C_YELLOW "alphabet ");
    e += print_code_list_diff(alphabet, a_len, target_alphabet, ta_len, a->utf8);
    PUT(C_YELLOW "collapsed");
    e += print_code_list_diff(collapse_chars, c_len, target_collapse_chars,
        tc_len, a->utf8);

    //XXX
    if ( 0 && a->first_char && a->first_char != 0){
      DEBUG(C_RED "collapse representative character should be %c, is %c" C_NORMAL,
          a->first_char, 0);
      e++;
    }
    if (e){
      errors++;
      DEBUG(C_REV_RED "Errors found!" C_NORMAL);
      PUT(C_BLUE "alphabet : " C_NORMAL);
      dump_alphabet(alphabet, a_len, a->utf8);
      PUT(C_DARK_CYAN "target   : " C_NORMAL);
      dump_alphabet(target_alphabet, ta_len, a->utf8);
      DEBUG("literal: %s", a->alphabet);
      PUT(C_BLUE "collapsed: " C_NORMAL);
      dump_alphabet(collapse_chars, c_len, a->utf8);
      PUT(C_DARK_CYAN "target   : " C_NORMAL);
      dump_alphabet(target_collapse_chars, tc_len, a->utf8);
      DEBUG("literal : %s", a->collapse);
    }
    DEBUG("--\n");
  }
  return errors;
}

int main(void){
  int r = 0;
  r += test_alphabet_finding();
  return r;
}
