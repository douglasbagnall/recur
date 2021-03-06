/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL

Initialisation functions for character-based models.
*/
#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include <math.h>
#include "path.h"
#include "badmaths.h"
#include <stdio.h>
#include "colour.h"
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
rnn_char_find_alphabet_s(const char *text, int len, RnnCharAlphabet *alphabet,
    double threshold, double digit_adjust, double alpha_adjust){
  int ignore_case = alphabet->flags & RNN_CHAR_FLAG_CASE_INSENSITIVE;
  int collapse_space = alphabet->flags & RNN_CHAR_FLAG_COLLAPSE_SPACE;
  int utf8 = alphabet->flags & RNN_CHAR_FLAG_UTF8;
  int n_chars = utf8 ? 0x200000 : 256;
  int *counts = calloc(n_chars + 1, sizeof(int));
  int c, prev = 0;
  int n = 0;
  int a_count = 0;
  int c_count = 0;

  const char *s = text;
  for(int i = 0; i < len; i++){
    if (s >= text + len){
      break;
    }
    if (utf8){
      c = read_utf8_char(&s);
      if (c < 0){
        if (i > 8){
          STDERR_DEBUG("Unicode Error at %d (%c %c %c %c %c %c %c %c)!", i,
              s[-7], s[-6], s[-5], s[-4], s[-3], s[-2], s[-1], s[0]);
        }
        else{
          STDERR_DEBUG("Unicode Error at %d!", i);
        }
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
    alphabet->points[0] = max_collapsed_point;
    counts[max_collapsed_point] = 0; /*so as to ignore it hereafter*/
    a_count = 1;
  }
  /*map the rest of the collapsed chars to alphabet->points[0]*/
  for (int i = 0; i < n_chars; i++){
    int count = counts[i];
    if (count){
      int adj_count = adjust_count(i, count, digit_adjust, alpha_adjust);
      if (adj_count >= min_count){
        if (a_count == 256){
          goto error;
        }
        alphabet->points[a_count] = i;
        a_count++;
      }
      else {
        if (c_count == 256){
          goto error;
        }
        alphabet->collapsed_points[c_count] = i;
        c_count++;
      }
    }
  }
  if (a_count == 0){
    goto error;
  }

  free(counts);
  alphabet->len = a_count;
  alphabet->collapsed_len = c_count;
  rnn_char_alphabet_set_flags(alphabet, ignore_case, utf8, collapse_space);

  DEBUG("alphabet len %i collapsed len %i", a_count, c_count);
  return 0;
 error:
  STDERR_DEBUG("threshold of %f over %d chars led to %d in alphabet, "
      "%d collapsed characters",
      threshold, n, a_count, c_count);
  free(counts);
  alphabet->len = 0;
  alphabet->collapsed_len = 0;
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
rnn_char_find_alphabet_f(const char *filename, RnnCharAlphabet *alphabet,
    double threshold, double digit_adjust, double alpha_adjust){
  int len;
  char *contents;
  int err = rnn_char_alloc_file_contents(filename, &contents, &len);
  if (! err){
    err = rnn_char_find_alphabet_s(contents, len, alphabet,
        threshold, digit_adjust, alpha_adjust);
    free(contents);
  }
  else {
    alphabet->len = 0;
    alphabet->collapsed_len = 0;
  }
  return err;
}

static inline int
find_space(const RnnCharAlphabet *alphabet)
{
  int i;
  for (i = 0; i < alphabet->len; i++){
    if (alphabet->points[i] == ' '){
      return i;
    }
  }
  DEBUG("space is not in alphabet; using collapse_target");
  return 0;
}


int*
rnn_char_new_char_lut(const RnnCharAlphabet *alphabet){
  int case_insensitive = alphabet->flags & RNN_CHAR_FLAG_CASE_INSENSITIVE;
  int i;
  int collapse_target = 0;
  int space = find_space(alphabet);
  int len = (alphabet->flags & RNN_CHAR_FLAG_UTF8) ? 0x200001 : 257;
  int *ctn = malloc(len *sizeof(int));
  /*anything unspecified goes to space */
  for (i = 0; i < len; i++){
    ctn[i] = space;
  }
  /*collapse chars map to alphabet->points[0] */
  for (i = 0; i < alphabet->collapsed_len; i++){
    int c = alphabet->collapsed_points[i];
    ctn[c] = collapse_target;
  }

  for (i = 0; i < alphabet->len; i++){
    int c = alphabet->points[i];
    ctn[c] = i;
    /*FIXME: case insensitivity works for ascii only */
    if (islower(c) && case_insensitive){
      ctn[toupper(c)] = i;
    }
  }
  return ctn;
}

/* Encode a text string in a given alphabet, returning
   new (u8*) memory */

u8 *
rnn_char_alloc_encoded_text(RnnCharAlphabet *alphabet, const char *text,
    int byte_len, int *encoded_len, int *char_to_net, bool verbose)
{
  int collapse_space = alphabet->flags & RNN_CHAR_FLAG_COLLAPSE_SPACE;
  int utf8 = alphabet->flags & RNN_CHAR_FLAG_UTF8;
  int i, j;
  int space = find_space(alphabet);
  u8 prev = space;
  u8 c;
  int chr = 0;
  int cleanup_char_to_net;
  if (char_to_net == NULL){
    char_to_net = rnn_char_new_char_lut(alphabet);
    cleanup_char_to_net = 1;
  }
  else {
    cleanup_char_to_net = 0;
  }

  u8 *encoded_text = malloc_aligned_or_die(byte_len * 2 + 4);

  const char *s = text;
  for(i = 0, j = 0; i < byte_len; i++){
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
        encoded_text[j] = c;
        j++;
      }
    }
    else {
      encoded_text[j] = c;
      j++;
    }
  }
  encoded_text[j] = 0;
  *encoded_len = j;
  if (cleanup_char_to_net){
    free(char_to_net);
  }
  if (verbose){
    STDERR_DEBUG("original text was %d chars (%d bytes), encoded is %d",
        i, byte_len, *encoded_len);
  }

  return realloc(encoded_text, j + 1);
}

/* Load and encode text from the named file, using the given alphabet.
   Returns new (u8*) memory */

u8*
rnn_char_load_new_encoded_text(const char *filename,
    RnnCharAlphabet *alphabet, int *encoded_len, int quietness)
{
  char *raw_text;
  int raw_len;
  bool verbose = (quietness < 1);

  rnn_char_alloc_file_contents(filename, &raw_text, &raw_len);

  u8 *encoded_text = rnn_char_alloc_encoded_text(alphabet,
      raw_text, raw_len, encoded_len, NULL, verbose);

  free(raw_text);
  return encoded_text;
}

void
rnn_char_dump_alphabet(RnnCharAlphabet *alphabet){
  char *s;
  char *s2;
  int utf8 = alphabet->flags & RNN_CHAR_FLAG_UTF8;

  s = new_string_from_codepoints(alphabet->points, alphabet->len, utf8);
  s2 = new_string_from_codepoints(alphabet->collapsed_points,
      alphabet->collapsed_len, utf8);

  DEBUG("alphabet:  " C_DARK_YELLOW "»»" C_NORMAL "%s" C_DARK_YELLOW "««" C_NORMAL, s);
  for (int i = 0; i < alphabet->len; i++){
    fprintf(stderr, "%d, ", alphabet->points[i]);
  }
  putc('\n', stderr);

  DEBUG("collapsed: " C_DARK_YELLOW "»»" C_NORMAL "%s" C_DARK_YELLOW "««" C_NORMAL, s2);
  for (int i = 0; i < alphabet->collapsed_len; i++){
    fprintf(stderr, "%d, ", alphabet->collapsed_points[i]);
  }
  putc('\n', stderr);

  free(s);
  free(s2);
}


RnnCharClassifiedChar *
rnn_char_alloc_classified_text(RnnCharClassBlock *b,
    RnnCharAlphabet *alphabet, int *text_len, int ignore_start){
  int collapse_space = alphabet->flags & RNN_CHAR_FLAG_COLLAPSE_SPACE;
  int utf8 = alphabet->flags & RNN_CHAR_FLAG_UTF8;
  int *char_to_net = rnn_char_new_char_lut(alphabet);
  int space = find_space(alphabet);
  int size = 1000 * 1000;
  RnnCharClassifiedChar *text = malloc(size * sizeof(RnnCharClassifiedChar));

  int len = 0;
  u8 prev = 0;
  u8 c = 0;
  for (; b; b = b->next){
    u8 class = b->class_code;
    const char *s = b->text;
    int chr = 0;
    int end = len + b->len;
    while (end > size){
      size *= 2;
      text = realloc(text, size * sizeof(RnnCharClassifiedChar));
    }
    for(int i = 0; len < end; i++){
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
        text[len].class = (i >= ignore_start) ? class : NO_CLASS;
        text[len].symbol = c;
        len++;
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


/*see also new_string_from_codepoints() et.al. in utf8.h */
char *
rnn_char_uncollapse_text(RnnCharAlphabet *alphabet, const u8 *orig, int len,
    int *dest_len)
{
  int i;
  int utf8 = alphabet->flags & RNN_CHAR_FLAG_UTF8;
  int mem_len = (len + 2) * (utf8 ? 4 : 1);
  char *mem = malloc(mem_len);
  char *s = mem;
  const int *lut = alphabet->points;
  for (i = 0; i < len; i++){
    u8 c = orig[i];
    int code = lut[c];
    if (code == 0){
      break;
    }
    if (utf8){
      int wrote = write_utf8_char(code, s);
      if (wrote == 0){
        STDERR_DEBUG("bad unicode code %d", code);
        break;
      }
      s += wrote;
    }
    else {
      *s = code;
      s++;
    }
  }
  *s = 0;
  s++;
  *dest_len = s - mem - 1;
  return realloc(mem, s - mem);
}




static inline char *
urlencode_alloc(const char *orig){
  size_t len = strlen(orig);
  char *s = malloc(len * 3 + 1);
  const char *lut = "0123456789abcdef";
  uint i, j;
  for (i = 0, j = 0; i < len; i++){
    char c = orig[i];
    if (c > 32 && c < 127 && c != '%'){
      s[j] = c;
      j++;
    }
    else {
      u8 u = (u8)c;
      s[j] = '%';
      s[j + 1] = lut[u >> 4];
      s[j + 2] = lut[u & 15];
      j += 3;
    }
  }
  s[j] = 0;
  return realloc(s, j + 1);
}

static inline char *
urldecode_alloc(const char *orig){
  size_t len = strlen(orig);
  char *s = malloc(len + 1);
  uint i, j;
  for (i = 0, j = 0; i < len; i++){
    char c = orig[j];
    if (c == '%'){
      char d;
      c = orig[j + 1];
      d = (((c & 0x40) ? c + 9 : c) & 15) << 4;
      c = orig[j + 2];
      d += ((c & 0x40) ? c + 9 : c) & 15;
      j += 3;
      s[i] = d;
    }
    else {
      s[i] = c;
      j++;
    }
  }
  s[i] = 0;
  return realloc(s, i + 1);
}


char *
rnn_char_construct_metadata(const struct RnnCharMetadata *m){
  char *metadata;
  char *enc_alphabet = urlencode_alloc(m->alphabet);
  char *enc_collapse_chars = urlencode_alloc(m->collapse_chars);

  int ret = asprintf(&metadata,
      "alphabet %s\n"
      "collapse_chars %s\n"
      "utf8 %d\n"
      "collapse_space %d\n"
      "case_insensitive %d\n"
      ,
      enc_alphabet,
      enc_collapse_chars,
      m->utf8,
      m->collapse_space,
      m->case_insensitive
  );
  if (ret == -1){
    FATAL_ERROR("can't alloc memory for metadata. or something.");
  }
  free(enc_alphabet);
  free(enc_collapse_chars);
  return metadata;
}

int
rnn_char_load_metadata(const char *orig, struct RnnCharMetadata *m){
  char *metadata = strdup(orig);
  char *key = NULL;
  char *value = NULL;
  char *value_end;
  char *s = metadata;
  DEBUG("Loading metadata\n%s", metadata);

#define get_val(k) do {                                           \
    key = strsep(&s, " ");                                        \
    value = strsep(&s, "\n");                                     \
    /*DEBUG("key is %s, k %s, value %s", key, k, value);   */     \
    if (!key || strcmp(key, (k))){                                \
      goto error;                                                 \
    }                                                             \
  } while(0)

#define get_int_val(k, dest) do {                       \
    get_val(k);                                         \
    dest = strtol(value, &value_end, 10);               \
    if (value == value_end){                            \
      DEBUG(k " value is missing or non-integer");      \
    }                                                   \
  } while(0)

      //key = strsep(&s, " ");

  get_val("alphabet");
  m->alphabet = urldecode_alloc(value);

  get_val("collapse_chars");
  m->collapse_chars = urldecode_alloc(value);

  get_int_val("utf8", m->utf8);
  get_int_val("collapse_space", m->collapse_space);
  get_int_val("case_insensitive", m->case_insensitive);

  if (s && *s){
    DEBUG("Found extra metadata: %s", s);
  }

  DEBUG("alphabet %s\n"
      "collapse_chars %s\n"
      "utf8 %d\n"
      "collapse_space %d\n"
      "case_insensitive %d\n"
      ,
      m->alphabet,
      m->collapse_chars,
      m->utf8,
      m->collapse_space,
      m->case_insensitive
  );

#undef get_val
#undef get_int_val

  free(metadata);
  return 0;
 error:
  DEBUG("Error loading metadata. key is %s, value %s", key, value);
  free(metadata);
  return -1;
}

void
rnn_char_free_metadata_items(struct RnnCharMetadata *m){
  free(m->alphabet);
  free(m->collapse_chars);
}

void
rnn_char_copy_metadata_items(struct RnnCharMetadata *src, struct RnnCharMetadata *dest){
  if (dest->alphabet){
    free(dest->alphabet);
  }
  if (dest->collapse_chars){
    free(dest->collapse_chars);
  }
  dest->alphabet = strdup(src->alphabet);
  dest->collapse_chars = strdup(src->collapse_chars);

  dest->utf8 = src->utf8;
  dest->collapse_space = src->collapse_space;
  dest->case_insensitive = src->case_insensitive;
}

char*
rnn_char_construct_net_filename(struct RnnCharMetadata *m, const char *basename,
    int input_size, int bottom_size, int hidden_size, int output_size){
  char s[260];
  char *metadata = rnn_char_construct_metadata(m);
  u32 sig = rnn_hash32(metadata);
  free(metadata);
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

int
rnn_char_check_metadata(RecurNN *net, struct RnnCharMetadata *m,
    bool trust_file_metadata, bool force_metadata){
  /*Check that the metadata struct matches the metadata serialisation in the
    net. If it doesn't the solution depends on the flags.
   */
  if (net == NULL || m == NULL){
    DEBUG("net is %p, metadata is %p, in %s", net, m, __func__);
    return -1;
  }
  int ret = 0;
  char *metadata = rnn_char_construct_metadata(m);
  if (net->metadata && strcmp(metadata, net->metadata)){
    DEBUG("metadata doesn't match. Expected:\n%s\nLoaded from net:\n%s\n",
        metadata, net->metadata);
    if (trust_file_metadata){
      /*this filename was specifically requested, so its metadata is
        presumably right. But first check if it even loads!*/
      struct RnnCharMetadata m2;
      int err = rnn_char_load_metadata(net->metadata, &m2);
      if (err){
        DEBUG("The net's metadata doesn't load."
            "Using otherwise determined metadata");
      }
      else {
        /*NB. the alphabet length could be different, isn't checked*/
        DEBUG("Using the net's metadata. Use --force-metadata to override");
        DEBUG("alphabet %s", m2.alphabet);
        rnn_char_copy_metadata_items(&m2, m);
        rnn_char_free_metadata_items(&m2);
      }
    }
    else if (force_metadata){
      DEBUG("Updating the net's metadata to match that requested "
          "(because --force-metadata)");
      free(net->metadata);
      net->metadata = strdup(metadata);
    }
    else {
      ret = -2;
    }
  }

  free(metadata);
  return ret;
}

RnnCharAlphabet *
rnn_char_new_alphabet(void){
  RnnCharAlphabet *a = malloc(sizeof(RnnCharAlphabet));
  a->points = calloc(257, sizeof(int));
  a->collapsed_points = calloc(257, sizeof(int));
  a->len = 0;
  a->collapsed_len = 0;
  a->flags = 0;
  return a;
}


RnnCharAlphabet
*rnn_char_new_alphabet_from_net(RecurNN *net){
  RnnCharMetadata m = {0};
  rnn_char_load_metadata(net->metadata, &m);
  RnnCharAlphabet *a = rnn_char_new_alphabet();
  rnn_char_alphabet_set_flags(a, m.case_insensitive, m.utf8, m.collapse_space);

  a->len = fill_codepoints_from_string(a->points, 256, m.alphabet, m.utf8);
  a->collapsed_len = fill_codepoints_from_string(a->collapsed_points, 256,
      m.collapse_chars, m.utf8);

  if (a->len != net->input_size || a->len != net->output_size){
    DEBUG("net sizes in %d out %d, alphabet length %d.",
        net->input_size, net->output_size, a->len);
  }
  return a;
}

void
rnn_char_alphabet_set_flags(RnnCharAlphabet *a,
    bool case_insensitive, bool utf8, bool collapse_space){
  /*XXX changing flags late could break things */
  a->flags = (
      (case_insensitive ? RNN_CHAR_FLAG_CASE_INSENSITIVE : 0) |
      (collapse_space   ? RNN_CHAR_FLAG_COLLAPSE_SPACE : 0) |
      (utf8             ? RNN_CHAR_FLAG_UTF8 : 0));
}

void
rnn_char_free_alphabet(RnnCharAlphabet *a){
  free(a->points);
  free(a->collapsed_points);
  free(a);
}

int
rnn_char_get_codepoint(RnnCharAlphabet *a, const char *s){
  int p = 0;
  fill_codepoints_from_string(&p, 1, s, a->flags & RNN_CHAR_FLAG_UTF8);
  for (int i = 0; i < a->len; i++){
    if (a->points[i] == p){
      return i;
    }
  }
  return -1;
}
