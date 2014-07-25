/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL

Initialisation functions for character-based models.
*/
#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include <math.h>
#include "path.h"
#include "badmaths.h"
#include <stdio.h>

#include "charmodel.h"
#include "utf8.h"

static inline ALWAYS_INLINE int
adjust_count(int i, int count, double digit_adjust, double alpha_adjust){
  if (count){
    if (i < 256){
      if (isdigit(i)){
        count = count * digit_adjust + 0.5;
      }
      else if (isalpha(i)){
        count = count * alpha_adjust + 0.5;
      }
    }
  }
  return count;
}

/* rnn_char_find_alphabet_s returns 0 for success, -1 on failure. */
int
rnn_char_find_alphabet_s(const char *text, int len, int *alphabet, int *a_len,
    int *collapse_chars, int *c_len, double threshold, double digit_adjust,
    double alpha_adjust, u32 flags){
  int ignore_case = flags & RNN_CHAR_FLAG_CASE_INSENSITIVE;
  int collapse_space = flags & RNN_CHAR_FLAG_COLLAPSE_SPACE;
  int utf8 = flags & RNN_CHAR_FLAG_UTF8;
  int n_chars = utf8 ? 0x200000 : 256;
  int *counts = calloc(n_chars + 1, sizeof(int));
  int c, prev = 0;
  int n = 0;
  const char *s = text;
  for(int i = 0; i < len; i++){
    if (s >= text + len){
      break;
    }
    if (utf8){
      c = read_utf8_char(&s);
      if (c < 0){
        STDERR_DEBUG("Unicode Error!");
        break;
      }
      else if (c == 0){
        break;
      }
    }
    else {
      c = ((u8*)s)[i];
    }
    if (c >= n_chars){
      DEBUG("got char %d, but there are only %d slots", c, n_chars);
      goto error;
    }

    if (c == 31){/* 31 is metadata separator XXX gah */
      c = 32;
    }
    if (collapse_space){
      if (isspace(c)){
        c = 32;
        if (c == prev){
          continue;
        }
      }
    }
    if (ignore_case && c < 0x80){
      /*FIXME ascii only */
      if(isupper(c)){
        c = tolower(c);
      }
    }
    n++;
    counts[c]++;
    prev = c;
  }
  if (n == 0){
    goto error;
  }
  int a_count = 0;
  int c_count = 0;

  /*find the representative for the collapsed_chars, if any, which is put at
    the beginning of the alphabet.*/
  int max_collapsed_count = 0;
  int max_collapsed_point = 0;
  int min_count = MAX(ceil(threshold * n), 1);
  DEBUG("min count %i threshold %f n %d", min_count, threshold, n);
  for (int i = 0; i < n_chars; i++){
    int count = counts[i];
    if (count){
      int adj_count = adjust_count(i, count, digit_adjust, alpha_adjust);
      /*select the representative on raw count, not adjusted count */
      if (adj_count < min_count && count > max_collapsed_count){
        max_collapsed_count = count;
        max_collapsed_point = i;
      }
    }
  }
  if (max_collapsed_count){
    alphabet[0] = max_collapsed_point;
    counts[max_collapsed_point] = 0; /*so as to ignore it hereafter*/
    a_count = 1;
  }
  /*map the rest of the collapsed chars to alphabet[0]*/
  for (int i = 0; i < n_chars; i++){
    int count = counts[i];
    if (count){
      int adj_count = adjust_count(i, count, digit_adjust, alpha_adjust);
      if (adj_count >= min_count){
        if (a_count == 256){
          goto error;
        }
        alphabet[a_count] = i;
        a_count++;
      }
      else {
        if (c_count == 256){
          goto error;
        }
        collapse_chars[c_count] = i;
        c_count++;
      }
    }
  }
  if (a_count == 0){
    goto error;
  }

  free(counts);
  *a_len = a_count;
  *c_len = c_count;
  DEBUG("a_len %i c_len %i", a_count, c_count);
  return 0;
 error:
  STDERR_DEBUG("threshold of %f over %d chars led to %d in alphabet, "
      "%d collapsed characters",
      threshold, n, a_count, c_count);
  free(counts);
  *a_len = *c_len = 0;
  return -1;
}

static inline long
get_file_length(FILE *f, int *err){
  long len = 0;
  *err = fseek(f, 0, SEEK_END);
  if (! *err){
    len = ftell(f);
    *err = fseek(f, 0, SEEK_SET);
  }
  return len;
}

int
rnn_char_alloc_file_contents(const char *filename, char **contents, int *len)
{
  FILE *f = fopen(filename, "r");
  int err;
  if (!f){
    goto early_error;
  }
  *len = get_file_length(f, &err);
  if (err){
    goto error;
  }
  char *c = malloc(*len + 4);
  if (! c){
    goto error;
  }
  int rlen = fread(c, 1, *len, f);
  if (rlen != *len){
    goto late_error;
  }
  fclose(f);
  c[*len] = 0; /*in case it gets used as string*/
  *contents = c;
  return 0;
 late_error:
  free(c);
 error:
  fclose(f);
 early_error:
  STDERR_DEBUG("could not read %s", filename);
  *contents = NULL;
  *len = 0;
  return -1;
}

/* rnn_char_find_alphabet_f returns 0 for success, -1 on failure. */
int
rnn_char_find_alphabet_f(const char *filename, int *alphabet, int *a_len,
    int *collapse_chars, int *c_len, double threshold, double digit_adjust,
    double alpha_adjust, u32 flags){
  int len;
  char *contents;
  int err = rnn_char_alloc_file_contents(filename, &contents, &len);
  if (! err){
    err = rnn_char_find_alphabet_s(contents, len, alphabet, a_len,
        collapse_chars, c_len, threshold, digit_adjust, alpha_adjust, flags);
    free(contents);
  }
  else {
    *a_len = *c_len = 0;
  }
  return err;
}


static int*
new_char_lut(const int *alphabet, int a_len, const int *collapse, int c_len,
    int *_space, u32 flags){
  int case_insensitive = flags & RNN_CHAR_FLAG_CASE_INSENSITIVE;
  int i;
  int collapse_target = 0;
  int space = 0;
  for (i = 0; i < a_len; i++){
    if (alphabet[i] == ' '){
      space = i;
      break;
    }
  }
  if (space == 0){
    DEBUG("space is not in alphabet; using collapse_target");
  }
  *_space = space;
  int len = (flags & RNN_CHAR_FLAG_UTF8) ? 0x200001 : 257;
  int *ctn = malloc(len *sizeof(int));
  /*anything unspecified goes to space */
  for (i = 0; i < len; i++){
    ctn[i] = space;
  }
  /*collapse chars map to alphabet[0] */
  for (i = 0; i < c_len; i++){
    int c = collapse[i];
    ctn[c] = collapse_target;
  }

  for (i = 0; i < a_len; i++){
    int c = alphabet[i];
    ctn[c] = i;
    /*FIXME: case insensitivity works for ascii only */
    if (islower(c) && case_insensitive){
      ctn[toupper(c)] = i;
    }
  }
  return ctn;
}

u8*
rnn_char_alloc_collapsed_text(char *filename, int *alphabet, int a_len,
    int *collapse_chars, int c_len, int *text_len, u32 flags, int quietness){
  int collapse_space = flags & RNN_CHAR_FLAG_COLLAPSE_SPACE;
  int utf8 = flags & RNN_CHAR_FLAG_UTF8;
  int i, j;
  int space;
  int *char_to_net = new_char_lut(alphabet, a_len,
      collapse_chars, c_len, &space, flags);
  u8 *text;
  int raw_len;
  rnn_char_alloc_file_contents(filename, (char**)&text, &raw_len);
  u8 prev = 0;
  u8 c;
  int chr = 0;
  const char *s = (char*)text;
  for(i = 0, j = 0; i < raw_len; i++){
    if (utf8){
      chr = read_utf8_char(&s);
      if (chr <= 0){
        break;
      }
    }
    else {
      chr = text[i];
      if (chr == 0)
        break;
    }
    c = char_to_net[chr];
    if (collapse_space){
      if (c != space || prev != space){
        prev = c;
        text[j] = c;
        j++;
      }
    }
    else {
      text[j] = c;
      j = i;
    }
  }
  text[j] = 0;
  *text_len = j;
  free(char_to_net);
  if (quietness < 1){
    STDERR_DEBUG("original text was %d chars (%d bytes), collapsed is %d",
        i, raw_len, *text_len);
  }
  return text;
}

RnnCharClassifiedChar *
rnn_char_alloc_classified_text(RnnCharClassBlock *b,
    int *alphabet, int a_len, int *collapse_chars, int c_len,
    int *text_len, u32 flags){
  int i;
  int space;
  int collapse_space = flags & RNN_CHAR_FLAG_COLLAPSE_SPACE;
  int utf8 = flags & RNN_CHAR_FLAG_UTF8;

  if (! (flags & RNN_CHAR_FLAG_UTF8)){
    DEBUG("WARNING: rnn_char_alloc_classified_text() only does utf-8 for now");
    flags |= RNN_CHAR_FLAG_UTF8;
  }
  int *char_to_net = new_char_lut(alphabet, a_len,
      collapse_chars, c_len, &space, flags);

  int size = 1000 * 1000;
  RnnCharClassifiedChar *text = malloc(size * sizeof(RnnCharClassifiedChar));

  int len = 0;
  u8 prev;

  for (; b; b = b->next){
    u8 class = b->class_code;
    const char *s = b->text;
    u8 c = 0;
    int chr = 0;
    int end = len + b->len;
    while (end > size){
      size *= 2;
      text = realloc(text, size * sizeof(RnnCharClassifiedChar));
    }
    for(; len < end; len++){
      if (utf8){
        chr = read_utf8_char(&s);
      }
      else {
        chr = (u8)*s;
        s++;
      }
      if (chr <= 0){
        break;
      }
      prev = c;
      c = char_to_net[chr];
      if (!(collapse_space && c == space && prev == space)){
        text[len].class = class;
        text[len].symbol = c;
        i++;
      }
    }
    if (chr < 0){
      DEBUG("seems like unicode trouble!");
    }
  }
  text = realloc(text, (len + 1) * sizeof(RnnCharClassifiedChar));
  *text_len = len;
  free(char_to_net);
  return text;
}

void
rnn_char_dump_collapsed_text(const u8 *text, int len, const char *name,
    const char *alphabet)
{
  int i;
  FILE *f = fopen_or_abort(name, "w");
  for (i = 0; i < len; i++){
    u8 c = text[i];
    fputc(alphabet[c], f);
  }
  fclose(f);
}

char *
rnn_char_construct_metadata(const struct RnnCharMetadata *m){
  char *metadata;
  int ret = asprintf(&metadata,
#define SEP "\x1F"
      "alphabet"       SEP "%s" SEP
      "collapse_chars" SEP "%s"
#undef SEP
      ,
      m->alphabet,
      m->collapse_chars
  );
  if (ret == -1){
    FATAL_ERROR("can't alloc memory for metadata. or something.");
  }
  return metadata;
}

int
rnn_char_load_metadata(const char *metadata, struct RnnCharMetadata *m){

  /*0x1f is the ascii field separator character.*/

#define CHECK_KEY(str, wanted) do {                                     \
    char * token = strtok(str, "\x1F");                                 \
    if (strcmp(token, wanted)){                                         \
      STDERR_DEBUG("looking for '%s', found '%s'", wanted, token);      \
      goto error;                                                       \
    }                                                                   \
  }while(0)                                                             \

  char *s = strdup(metadata);
  CHECK_KEY(s, "alphabet");
  m->alphabet = strdup(strtok(NULL, "\x1F"));
  CHECK_KEY(s, "collapse_chars");
  m->collapse_chars = strdup(strtok(NULL, "\x1F"));

#undef CHECK_KEY

  free(s);
  return 0;
 error:
  return 1;
}

void
rnn_char_free_metadata_items(struct RnnCharMetadata *m){
  free(m->alphabet);
  free(m->collapse_chars);
}

char*
rnn_char_construct_net_filename(struct RnnCharMetadata *m, const char *basename,
    int alpha_size, int bottom_size, int hidden_size){
  char s[260];
  char *metadata = rnn_char_construct_metadata(m);
  int input_size = alpha_size;
  int output_size = alpha_size;
  u32 sig = rnn_hash32(metadata);
  if (bottom_size){
    snprintf(s, sizeof(s), "%s-s%0" PRIx32 "-i%d-b%d-h%d-o%d.net", basename,
        sig, input_size, bottom_size, hidden_size, output_size);
  }
  else{
    snprintf(s, sizeof(s), "%s-s%0" PRIx32 "-i%d-h%d-o%d.net", basename,
        sig, input_size, hidden_size, output_size);
  }
  DEBUG("filename: %s", s);
  return strdup(s);
}
