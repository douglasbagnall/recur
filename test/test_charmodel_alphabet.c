/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL
-*- coding: utf-8 -*-

Tests alphabet finding in ../charmodel-init.c
*/
#include "path.h"
#include "../charmodel.h"
#include "../utf8.h"
#include "../colour.h"

#define BREAK_ON_ERROR 1

#define EREWHON_TEXT TEST_DATA_DIR "/erewhon.txt"
#define LGPL_TEXT TEST_DATA_DIR "/../licenses/LGPL-2.1"
#define WAI1874_TEXT TEST_DATA_DIR "/Wai1874NgaM-nfc.txt"
#define WAI1874_NFD_TEXT TEST_DATA_DIR "/Wai1874NgaM-nfd.txt"

#define PUT(format, ...) fprintf(stderr, (format),## __VA_ARGS__)

typedef struct {
  double threshold;
  const char *alphabet;
  const char *collapse;
  const char first_char;
  char *filename;
  int ignore_case;
  int utf8;
  int collapse_space;
  double digit_adjust;
  double alpha_adjust;
} ab_test;

//Erewhon chars: note space at start:
// etaonihsrdlucmwfygpb,v.k-;x"qj'?:z)(_1!0*872&{}695/34[]@
static const ab_test ab_test_cases[] = {
  {
    .threshold = 3e-4,
    .alphabet = "z etaonihsrdlucmwfygpb,v.k-;x\"qj'?:",
    .collapse = ")(_1!0*872&{}695/34[]@",
    .first_char = 'z',
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
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
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
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
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
    .filename = EREWHON_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },
  { /*digit adjust */
    .threshold = 3e-5,
    .alphabet = "1 etaonihsrdlucmwfygpb,v.k-;x\"qj'?:z)(_!*&",
    .collapse = "{}0872695/34[]@",
    .first_char = '{',
    .digit_adjust = 0.3,
    .alpha_adjust = 1.0,
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
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
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
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
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
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
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
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
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
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
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
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
    .filename = LGPL_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },
  {/*LGPL text, digit adjust */
    .threshold = 1e-4,
    .alphabet = "2 etiorasnhlcduyfbpmwg,v.k)\"x1(q;j-/':><",
    .collapse = "09634587![]z`",
    .first_char = '6',
    .digit_adjust = 0.1,
    .alpha_adjust = 1.0,
    .filename = LGPL_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },
  {/*LGPL text, alpha, digit adjust */
    .threshold = 1e-4,
    .alphabet = "2 etiorasnhlcduyfbpmwg,v.k)\"x1(q;j-/':><z",
    .collapse = "06934587![]`",
    .first_char = '6',
    .digit_adjust = 0.1,
    .alpha_adjust = 3.0,
    .filename = LGPL_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },

  {/*Wai1874 text, UTF-8 NFC*/
    .threshold = 1e-4,
    .alphabet = "' aiteokhrnu.mgpw<>,1-0£sd42₤367859:)(;ā—v\"c&bjē*/l",
    .collapse = "…yxīōü",
    .first_char = '\'',
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
    .filename = WAI1874_TEXT,
    .ignore_case = 1,
    .utf8 = 1,
    .collapse_space = 1
  },

  {/*Wai1874 text, UTF-8 NFD (decomposed)*/
    .threshold = 1e-4,
    .alphabet = "' aiteokhrnu.mgpw<>,1-0£sd42₤367859:)(;—v\"c&bj*/l\u0304",
    .collapse = "…yx\u0308",
    .first_char = '\'',
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
    .filename = WAI1874_NFD_TEXT,
    .ignore_case = 1,
    .utf8 = 1,
    .collapse_space = 1
  },

  {/*Wai1874 text, ignore case */
    .threshold = 1e-4,
    .alphabet = "' aietoknrh.ugmp<>Kw,1MTH-W0RPN£sd42A₤36I785OE9:)(;ā—\"vUVcB&JlS*/ē",
    .collapse = "yD…xüXōCGī",
    .first_char = '\'',
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
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
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
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
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
    .filename = WAI1874_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },
  {/*utf-8 parsed as bytes, push down digits*/
    .threshold = 1e-4,
    .alphabet = ("1 aiteokhrnu.mgpw><,-\xa3\xc2s\xe2""d\xa4"
        ":\xc4();\x80\x81\x82v\x94\"c&bj*/\x93l"),
    .collapse = "\xa6\x8dxy\xc3\xbc\xc5\xab""234567890'",
    .first_char = '1',
    .digit_adjust = 0.01,
    .alpha_adjust = 1.0,
    .filename = WAI1874_TEXT,
    .ignore_case = 1,
    .utf8 = 0,
    .collapse_space = 1
  },

  {/*nfd utf-8 parsed as bytes, push down digits, raise letters*/
    .threshold = 1e-4,
    .alphabet = ("1 aiteokhrnu.mgpw><,-\xa3\xc2s\xe2""d\xa4"
        ":\xcc();\x80\x82v\x94\"c&bj*/\x84lxy"),
    .collapse = "\xa6\x88""234567890'",
    .first_char = '1',
    .digit_adjust = 0.01,
    .alpha_adjust = 2.0,
    .filename = WAI1874_NFD_TEXT,
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
    .digit_adjust = 1.0,
    .alpha_adjust = 1.0,
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

static int
test_alphabet_finding(void){
  int i;
  int errors = 0;

  for (i = 0; ; i++){
    const ab_test *a = &ab_test_cases[i];
    if (! a->filename){
      break;
    }
    RnnCharAlphabet *alphabet = rnn_char_new_alphabet();
    RnnCharAlphabet *target = rnn_char_new_alphabet();
    rnn_char_alphabet_set_flags(alphabet,
        a->ignore_case,
        a->utf8,
        a->collapse_space);

    int err = rnn_char_find_alphabet_f(a->filename, alphabet,
        a->threshold, a->digit_adjust, a->alpha_adjust);
    if (err){
      errors++;
      continue;
    }
    DEBUG(C_CYAN "%s" C_NORMAL " threshold %f, case %s, %s, %s space "
        "%sdigit adj %g %salpha adj %g" C_NORMAL,
        a->filename, a->threshold,
        a->ignore_case ? "insensitive" : "sensitive",
        a->utf8 ? "utf8" : "bytes",
        a->collapse_space ? "collapsed" : "preserved",
        a->digit_adjust == 1.0 ? C_NORMAL : C_YELLOW,
        a->digit_adjust,
        a->alpha_adjust == 1.0 ? C_NORMAL : C_YELLOW,
        a->alpha_adjust
    );

    if (a->utf8){
      target->len = fill_codepoints_from_utf8(target->points, 256, a->alphabet);
      target->collapsed_len = fill_codepoints_from_utf8(target->collapsed_points,
          256, a->collapse);
    }
    else {
      target->len = fill_codepoints_from_bytes(target->points, 256, a->alphabet);
      target->collapsed_len = fill_codepoints_from_bytes(target->collapsed_points,
          256, a->collapse);
    }
    int e = 0;
    DEBUG(C_GREEN "green" C_NORMAL ": in alphabet and target. "
        C_RED "red" C_NORMAL ": in found alphabet only. "
        C_MAGENTA "magenta" C_NORMAL ": in target only.");

    PUT(C_YELLOW "alphabet ");
    e += print_code_list_diff(alphabet->points, alphabet->len, target->points,
        target->len, a->utf8);
    PUT(C_YELLOW "collapsed");
    e += print_code_list_diff(alphabet->collapsed_points, alphabet->collapsed_len,
        target->collapsed_points, target->collapsed_len, a->utf8);

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
      rnn_char_dump_alphabet(alphabet);
      PUT(C_DARK_CYAN "target   : " C_NORMAL);
      rnn_char_dump_alphabet(target);
      DEBUG("literal alphabet:  %s", a->alphabet);
      DEBUG("literal collapsed: %s", a->collapse);
      if (BREAK_ON_ERROR)
        exit(1);
    }
    rnn_char_free_alphabet(alphabet);
    rnn_char_free_alphabet(target);
    DEBUG("--\n");
  }
  return errors;
}

int main(void){
  int r = 0;
  r += test_alphabet_finding();
  return r;
}
