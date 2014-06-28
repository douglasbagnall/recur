#include "../utf8.h"

#define C_NORMAL  "\033[00m"
#define C_DARK_GREEN  "\033[00;32m"
#define C_GREEN  "\033[01;32m"
#define C_DARK_RED  "\033[00;31m"
#define C_RED "\033[01;31m"
#define C_MAGENTA  "\033[01;35m"
#define C_PURPLE  "\033[00;35m"
#define C_CYAN  "\033[01;36m"
#define C_YELLOW  "\033[01;33m"
#define C_WHITE  "\033[01;37m"

static inline void
dump_int_array(const int* a, int len){
  for (int i = 0; i < len - 1; i++){
    fprintf(stderr, "%d, ", a[i]);
  }
  if (len){
    fprintf(stderr, "%d\n", a[len - 1]);
  }
}

static inline void
dump_byte_array(const char* a, int len){
  for (int i = 0; i < len - 1; i++){
    fprintf(stderr, "%u, ", (u8)a[i]);
  }
  if (len){
    fprintf(stderr, "%u\n", (u8)a[len - 1]);
  }
}

static const char *round_trip_strings[] = {
  "hello",
  "hēllo",
  "ā\n\t¬à⅜£ə⋄þÞøåơị̂·∴äċ ē¶ð…»“”⌷`wsΩąá-Āāēōī←",
  "顧己及人，請勿隨地大小二便", /*from http://languagelog.ldc.upenn.edu/nll/?p=13140*/

  NULL
};

static const char *bad_utf8_strings[] = {
  "should fail, bare continuation \x83",
  "incomplete character \xF3\x82 mid-string",
  "incomplete character at end \xF7\x9F",
  "\x89\x89\x89\x98\xa8\x89\x89\xc8\xd8\xd9 garbage at start",

  NULL
};

const int MAXLEN = 1000;

int
test_utf8_should_fail()
{
  int i;
  int errors = 0;
  int codepoints[MAXLEN + 1];
  for (i = 0; ; i++){
    const char *s = bad_utf8_strings[i];
    if (!s){
      break;
    }
    DEBUG("\nlooking at    \"%s\"", s);
    int n_points = fill_codepoints_from_string(codepoints, MAXLEN, s);
    dump_int_array(codepoints, n_points + 1);
    const char *s2 = new_string_from_codepoints(codepoints, n_points);
    int diff = strcmp(s, s2);
    if (! diff){
      errors++;
      DEBUG(C_RED "reconstruction works. it should fail" C_NORMAL);
    }
    else {
      DEBUG(C_GREEN "reconstruction rightly fails: " C_NORMAL
          "\"%s\"\noriginal len %lu, reconstructed %lu",
          s2, strlen(s), strlen(s2));
    }

    free((void*)s2);
  }
  return errors;
}



int
test_codepoint_round_trip(int(str2cp)(int*, int, const char*),
    char *(cp2new_str)(const int*, int))
{
  int i;
  int errors = 0;
  int codepoints[MAXLEN + 1];
  for (i = 0; ; i++){
    const char *s = round_trip_strings[i];
    if (!s){
      break;
    }
    int n_chars = strlen(s);
    DEBUG("\noriginal:      %s", s);
    int n_points = str2cp(codepoints, MAXLEN, s);
    dump_int_array(codepoints, n_points);
    const char *s2 = cp2new_str(codepoints, n_points);
    int diff = strcmp(s, s2);
    if (diff){
      errors++;
      int n_chars2 = strlen(s2);
      DEBUG(C_RED "reconstruction differs. len is %d" C_NORMAL, n_chars2);
      DEBUG("reconstructed: %s", s2);
      int n_points2 = str2cp(codepoints, MAXLEN, s2);
      dump_int_array(codepoints, n_points2);
      DEBUG("orignal bytes");
      dump_byte_array(s, n_chars);
      DEBUG("reconstructed bytes");
      dump_byte_array(s2, n_chars2);
    }
    DEBUG("points: %d, bytes: %d, %s diff: %d" C_NORMAL, n_points, n_chars,
        diff ? C_RED : C_GREEN, diff);
    free((void*)s2);
  }
  return errors;
}

int
main(void){
  int r = 0;

  DEBUG("\n" C_YELLOW "utf-8 reconstruction cycle" C_NORMAL);
  r += test_codepoint_round_trip(fill_codepoints_from_string,
      new_string_from_codepoints);

  DEBUG("\n" C_YELLOW "8 bit reconstruction" C_NORMAL);
  r += test_codepoint_round_trip(fill_codepoints_from_8bit_string,
      new_8bit_string_from_ints);

  DEBUG("\n" C_YELLOW "known bad strings where reconstruction should fail" C_NORMAL);
  r += test_utf8_should_fail();

  if (r){
    DEBUG(C_DARK_RED "\nTERRIBLE NEWS! " C_MAGENTA "%d "
        C_RED "failing tests" C_NORMAL, r);
  }
  else{
    DEBUG("\n" C_CYAN "All tests pass. Perhaps there need to be more." C_NORMAL);
  }
  return r;
}
