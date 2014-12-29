/* Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL

Helper functions to convert UTF-8 sequences int unicode codepoints, and vice
versa.

Obviously it should have been easy to use a library, but I was interested in
how it would work.
*/
#include "recur-common.h"
#include <stdbool.h>

/*write_escaped_char*/
static inline int
write_escaped_char(uint code, char *s){
  u8 x = code & 0xff;
  if (code > 126 || code < 32 || code == '\\' || code == '"'){
    snprintf(s, 5, "\\x%02x", x);
    return 4;
  }
  s[0] = code;
  return 1;
}

/*write_utf8_char(uint code, char *s) writes between 0 and 4 bytes at *s,
  returning the number of bytes written. 0 corresponds to a unicode
  codepoint that can't be represented in 4 bytes or fewer.
 */
static inline int
write_utf8_char(uint code, char *s){
  if (code < 0x80){
    *s = code;
    return 1;
  }
  if (code < 0x800){
    s[0] = 0xC0 | (code >> 6);
    s[1] = 0x80 | (code & 63);
    return 2;
  }
  if (code < 0x10000){
    s[0] = 0xE0 | (code >> 12);
    s[1] = 0x80 | ((code >> 6) & 63);
    s[2] = 0x80 | (code & 63);
    return 3;
  }
  if (code < 0x200000){
    s[0] = 0xF0 | (code >> 18);
    s[1] = 0x80 | ((code >> 12) & 63);
    s[2] = 0x80 | ((code >> 6) & 63);
    s[3] = 0x80 | (code & 63);
    return 4;
  }
  return 0;
}

static inline int
fput_utf8_char(uint code, FILE *f){
  char s[5];
  int n = write_utf8_char(code, s);
  s[n] = 0;
  if (fputs(s, f) == EOF){
    return EOF;
  }
  return n;
}

static inline ALWAYS_INLINE int
_parse_utf8_byte(int c, int *extra_bytes)
{
  if (! (c & 0x80)){
    *extra_bytes = 0;
  }
  else if ((c & 0xE0) == 0xC0){
    c &= 31;
    *extra_bytes = 1;
  }
  else if ((c & 0xF0) == 0xE0){
    c &= 15;
    *extra_bytes = 2;
  }
  else if ((c & 0xF8) == 0xF0){
    c &= 7;
    *extra_bytes = 3;
  }
  else if ((c & 0xC0 ) == 0x80){
    //stray continuation
    *extra_bytes = 0;
    return -1;
  }
  else {
    //super-high codepoint
    *extra_bytes = 0;
    return -2;
  }
  return c;
}

static inline ALWAYS_INLINE int
check_utf8_bounds(int c, int extra){
  /* For stricter compliance check for codepoints sneakily encoded with
     too many extra characters. Ranges are:

     1 -> 0x000080 - 0x0007ff    8 - 11 bits
     2 -> 0x000800 - 0x00ffff   12 - 16 bits
     3 -> 0x010000 - 0x1fffff   17 - 21 bits

     so the required function maps 1, 2, 3 to 1 << 7, 1 << 11, 1 << 16
  */
  int min = 1 << (1 + extra * 5 + (extra == 1));
  if (c < min){
    return -1;
  }
  return c;
}


/*read_utf8_char returns the unicode code point indicated by the UTF-8
  sequence starting at *s, and advances *s to beyond the character.

  Returns -1 if the UTF-8 is not valid, and -2 if it seems to define a
  codepoint on a very high plane (such as aren't actually used).

  The string pointer is never advanced by more than 4 characters.
*/
static inline int
read_utf8_char(const char **s){
  int c = **s;
  int extra_bytes;
  (*s)++;
  c = _parse_utf8_byte(c, &extra_bytes);
  if (extra_bytes){
    for (int i = 0; i < extra_bytes; i++){
      int x = **s;
      (*s)++;
      if ((x & 0xC0) != 0x80){
        //bad codepoint (perhaps end of string)
        return -1;
      }
      c <<= 6;
      c += x & 63;
    }
    c = check_utf8_bounds(c, extra_bytes);
  }
  return c;
}

/*fread_utf8_char() is like read_utf8_char(), except it takes a FILE*
  pointer.

  It returns 0 for EOF if it occurs between characters, but -1 if it occurs in
  what should be the middle of a UTF-8 character. Otherwise as read_utf8_char().
*/
static inline int
fread_utf8_char(FILE *f){
  int c = fgetc(f);
  int extra_bytes;
  if (c == EOF){
    return 0;
  }
  c = _parse_utf8_byte(c, &extra_bytes);
  if (extra_bytes){
    for (int i = 0; i < extra_bytes; i++){
      int x = fgetc(f);
      if (x == EOF || (x & 0xC0) != 0x80){
        MAYBE_DEBUG("UTF-8 stream seems to stop mid-character");
        return -1;
      }
      c <<= 6;
      c += x & 63;
    }
    c = check_utf8_bounds(c, extra_bytes);
  }
  return c;
}

static inline int
approx_isspace(int c){
  return (c < 33 || c == 160 || c == 0x180E ||
      (c >= 0x2000 && c <= 0x200b) ||
      c == 0x202f || c == 0x205f || c == 0x3000);
}

static inline char *
new_utf8_from_codepoints(const int *points, int maxlen){
  int i;
  char *str = malloc(maxlen * 4 + 1);
  char *s = str;
  for (i = 0; i < maxlen; i++){
    int code = points[i];
    int wrote = write_utf8_char(code, s);
    if (wrote == 0){
      STDERR_DEBUG("bad unicode code %d", code);
      break;
    }
    s += wrote;
  }
  *s = 0;
  s++;
  return realloc(str, s - str);
}

static inline char *
new_bytes_from_codepoints(const int *points, int maxlen){
  int i;
  char *str = malloc(maxlen + 1);
  for (i = 0; i < maxlen; i++){
    int c = points[i];
    if (! c){
      break;
    }
    str[i] = c;
  }
  str[i] = 0;
  return str;
}

static inline char *
new_string_from_codepoints(const int *points, int maxlen, bool utf8){
  if (utf8){
    return new_utf8_from_codepoints(points, maxlen);
  }
  return new_bytes_from_codepoints(points, maxlen);
}

static inline int
fill_codepoints_from_bytes(int *points, int len, const char *string){
  int i;
  const u8* s = (u8*)string;
  for (i = 0; i < len; i++){
    uint x = s[i];
    points[i] = x;
    if (! points[i]){
      break;
    }
  }
  return i;
}

static inline int
fill_codepoints_from_utf8(int *points, int len, const char *string){
  int i;
  const char **s = &string;
  for (i = 0; i < len; i++){
    int c = read_utf8_char(s);
    if (c <= 0){
      break;
    }
    points[i] = c;
  }
  return i;
}

static inline int
fill_codepoints_from_string(int *points, int len, const char *string, bool utf8){
  if (utf8){
    return fill_codepoints_from_utf8(points, len, string);
  }
  return fill_codepoints_from_bytes(points, len, string);
}
