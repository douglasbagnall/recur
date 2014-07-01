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
#define LGPL_TEXT "../licenses/LGPL-2.1"

typedef struct {
  double threshold;
  char *alphabet;
  char *collapse;
  char first_char;
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


  {
    .filename = NULL
  }
};

static inline int
print_char_list_diff(const char *a, const char *b)
{
  int i, j;
  char *b2 = strdupa(b);
  int diff = 0;
  fprintf(stderr, C_NORMAL "-->");
  for (i = 0; a[i]; i++){
    char x = a[i];
    int found = 0;
    for (j = 0; b[j]; j++){
      if (b[j] == x){
        b2[j] = 0;
        found = 1;
        break;
      }
    }
    if (found){
      fprintf(stderr, C_GREEN "%c", x);
    }
    else {
      fprintf(stderr, C_RED "%c", x);
      diff++;
    }
  }
  fprintf(stderr, C_MAGENTA);
  for (j = 0; b[j]; j++){
    if (b2[j]){
      fprintf(stderr, "%c", b2[j]);
      diff++;
    }
  }
  fprintf(stderr, C_NORMAL "<--diff is %d\n", diff);
  return diff;
}


static int
test_alphabet_finding(void){
  int i;
  int errors = 0;
  int *alphabet = malloc(257 * sizeof(int));
  int *collapse_chars = malloc(257 * sizeof(int));
  int a_len, c_len;
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
    char *a_string;
    char *c_string;
    if (a->utf8){
      a_string = new_string_from_codepoints(alphabet, a_len);
      c_string = new_string_from_codepoints(collapse_chars, c_len);
    }
    else {
      a_string = new_8bit_string_from_ints(alphabet, a_len);
      c_string = new_8bit_string_from_ints(collapse_chars, c_len);
    }
    int e = 0;
    fprintf(stderr, C_YELLOW "alphabet ");
    e += print_char_list_diff(a_string, a->alphabet);
    fprintf(stderr, C_YELLOW "collapsed");
    e += print_char_list_diff(c_string, a->collapse);
    if (a->first_char && a->first_char != *a_string){
      DEBUG(C_RED "collapse representative character should be %c, is %c" C_NORMAL,
          a->first_char, *a_string);
      e++;
    }
    if (e){
      errors++;
      DEBUG(C_REV_RED "Errors found!" C_NORMAL);
      DEBUG("alphabet: %s", a_string);
      DEBUG("target  : %s", a->alphabet);
      DEBUG("collapse: %s", c_string);
      DEBUG("target  : %s", a->collapse);
    }
    free(a_string);
    free(c_string);
    DEBUG("--\n");
  }
  return errors;
}

int main(void){
  int r = 0;
  r += test_alphabet_finding();
  return r;
}
