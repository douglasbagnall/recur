#include "recur-context.h"
#include "recur-common.h"
#include "rescale.h"
#include <string.h>
#include <stdio.h>
#include <ctype.h>


static inline void
consolidate_float_row(const float *restrict src, float *restrict dest,
    int src_len, int dest_len, int x_step, int n_rows){
  int i, j;
  uint acc = x_step / 2;
  float sum = 0;
  uint n_samples = 0;
  j = 0;
  for (i = 0; i < src_len; i++){
    if (acc >= 0x20000){
      dest[j] = sum / n_samples;
      j++;
      acc -= 0x20000;
      sum = 0;
      n_samples = 0;
    }
    sum += src[i];
    n_samples += n_rows;
    acc += x_step;
  }
  if (j < dest_len && n_samples){
    dest[j] = sum / n_samples;
  }
}

/*XXX cblas would do it, but overhead is probably big */
static inline void
exact_sum_row_float(const float *restrict src, float *restrict dest, int len){
  int i;
  for (i = 0; i < len; i++){
    dest[i] += src[i];
  }
}

void
recur_float_downscale(const float *restrict src, const int s_width,
    const int s_height, const int s_stride,
    float *restrict dest, const int d_width,
    const int d_height, const int d_stride)
{
  uint acc, n_rows;
  int y_step;
  int x_step;
  int y;
  y_step = 0x20000 * d_height / s_height;
  x_step = 0x20000 * d_width / s_width;
  float tmp_row[s_width];
  memset(tmp_row, 0, s_width * sizeof(float));
  acc = y_step / 2;
  const float *src_row = src;
  float *dest_row = dest;
  n_rows = 0;
  for (y = 0; y < s_height; y++) {
    if (acc >= 0x20000){
      consolidate_float_row(tmp_row, dest_row, s_width, d_width, x_step, n_rows);
      memset(tmp_row, 0, s_width * sizeof(float));
      acc -= 0x20000;
      dest_row += d_stride;
      n_rows = 0;
    }
    exact_sum_row_float(src_row, tmp_row, s_width);
    acc += y_step;
    src_row += s_stride;
    n_rows++;
  }
  if (dest_row <= dest + d_stride * (d_height - 1)){
    consolidate_float_row(tmp_row, dest_row, s_width, d_width, x_step, n_rows);
  }
}



static inline void
exact_sum_row(const u8 *restrict src, u16 *restrict dest, int len){
  int i;
  for (i = 0; i < len; i++){
    dest[i] += src[i];
  }
}

static inline void
consolidate_exact_row(const u16 *restrict src, u8 *restrict dest,
    int src_len, int dest_len, int x_step, int n_rows){
  int i, j;
  uint acc = x_step / 2;
  uint sum = 0;
  uint n_samples = 0;
  j = 0;
  for (i = 0; i < src_len; i++){
    if (acc >= 0x20000){
      sum += (n_samples / 2);
      dest[j] = sum / n_samples;
      j++;
      acc -= 0x20000;
      sum = 0;
      n_samples = 0;
    }
    sum += src[i];
    n_samples += n_rows;
    acc += x_step;
  }
  if (j < dest_len && n_samples){
    sum += (n_samples / 2);
    dest[j] = sum / n_samples;
  }
}

void
recur_exact_downscale(const u8 *restrict src, const int s_width,
    const int s_height, const int s_stride,
    u8 *restrict dest, const int d_width,
    const int d_height, const int d_stride){

  uint acc, n_rows;
  int y_step;
  int x_step;
  int y;
  y_step = 0x20000 * d_height / s_height;
  x_step = 0x20000 * d_width / s_width;

  u16 tmp_row[s_width];
  memset(tmp_row, 0, sizeof(tmp_row));
  acc = y_step / 2;
  const u8 *src_row = src;
  u8 *dest_row = dest;
  n_rows = 0;
  for (y = 0; y < s_height; y++) {
    if (acc >= 0x20000){
      consolidate_exact_row(tmp_row, dest_row, s_width, d_width, x_step, n_rows);
      memset(tmp_row, 0, sizeof(tmp_row));
      acc -= 0x20000;
      dest_row += d_stride;
      n_rows = 0;
    }
    exact_sum_row(src_row, tmp_row, s_width);
    acc += y_step;
    src_row += s_stride;
    n_rows++;
  }
  if (dest_row <= dest + d_stride * (d_height - 1)){
    consolidate_exact_row(tmp_row, dest_row, s_width, d_width, x_step, n_rows);
  }
}



static inline void
skipping_sum_row(const u8 *restrict src8, u64 *dest64, int len8){
  int i;
  u64 *src64 = (u64 *)src8; /*XXX wil this work unaligned?*/
  int len64 = len8 / sizeof(u64);

  for (i = 0; i < len64; i++){
    dest64[i] += src64[i] & 0x00ff00ff00ff00ff;
  }

  /*clean up the rest with 16 bit steps*/
  u16 *dest16 = (u16 *)dest64;
  u16 *src16 = (u16 *)src64;
  int len16 = len8 / sizeof(u16);
  for (i = len64 * sizeof(u64) / sizeof(u16); i < len16; i++){
    dest16[i] += src16[i] &0x00ff;
  }
}

static inline void
consolidate_skipped_row(const u64 *restrict src64, u8 *restrict dest8,
    int src_len, int dest_len, int x_step, int n_rows){
  int i, j;
  int acc = x_step / 4;
  int len16 = src_len / sizeof(u16);
  u16 *src16 = (u16 *)src64;
  int sum = 0;
  int n_samples = 0;
  j = 0;
  for (i = 0; i < len16; i++){
    if (acc >= 0x20000){
      sum += (n_samples / 2);
      dest8[j] = sum / n_samples;
      j++;
      acc -= 0x20000;
      sum = 0;
      n_samples = 0;
    }
    sum += src16[i];
    n_samples += n_rows;
    acc += x_step;
  }
  if (j < dest_len){
    sum += (n_samples / 2);
    dest8[j] = sum / n_samples;
  }
}

void
recur_skipping_downscale(const u8 *restrict src, const int s_width,
    const int s_height, const int s_stride,
    u8 *restrict dest, const int d_width,
    const int d_height, const int d_stride){
  uint acc, n_rows;
  int y;
  const int y_step = 0x20000 * 2 * d_height / s_height;
  const int x_step = 0x20000 * 2 * d_width / s_width;

  u64 tmp_row[s_width / sizeof(u64) + 1];
  memset(tmp_row, 0, sizeof(tmp_row));
  acc = y_step / 4;
  n_rows = 0;
  const u8 *src_row = src;
  u8 *dest_row = dest;
  for (y = 0; y < s_height; y += 2) {
    if (acc >= 0x20000){
      consolidate_skipped_row(tmp_row, dest_row, s_width, d_width, x_step, n_rows);
      memset(tmp_row, 0, sizeof(tmp_row));
      acc -= 0x20000;
      n_rows = 0;
      dest_row += d_stride;
    }
    skipping_sum_row(src_row, tmp_row, s_width);
    acc += y_step;
    src_row += s_stride * 2;
    n_rows++;
  }
  if (dest_row <= dest + d_stride * (d_height - 1)){
    consolidate_skipped_row(tmp_row, dest_row, s_width, d_width, x_step, n_rows);
  }
}


void
recur_adaptive_downscale(const u8 *src, const int s_width,
    const int s_height, const int s_stride,
    u8 *dest, const int d_width,
    const int d_height, const int d_stride){
  if (s_width >= d_width * 4 && s_height >= d_height * 4){
    recur_skipping_downscale(src, s_width, s_height, s_stride,
        dest, d_width, d_height, d_stride);
  }
  else {
    recur_exact_downscale(src, s_width, s_height, s_stride,
        dest, d_width, d_height, d_stride);
  }
}

Image *
recur_load_pgm_file(const char *filename){
  const int MAX_DIGITS = 9;
  char line[MAX_DIGITS + 3];
  FILE *f = fopen(filename, "rb");
  if (f == NULL){
    goto early_escape;
  }
  char *s;
  s = fgets(line, sizeof("P5"), f);
  if (s == NULL || strcmp(line, "P5")){
    goto escape;
  }
  int i, j = 0;
  uint dimensions[3];
  int c;
  for (i = 0; i < 3;){
    c = getc(f);
    if (c == '#'){
      while (c != '\n' && c != EOF)
        c = getc(f);
      if (c == EOF)
        goto escape;
    }
    if (isdigit(c) && j < MAX_DIGITS){
      line[j] = c;
      j++;
    }
    else if (isspace(c)){
      if (j){
        line[j] = 0;
        dimensions[i] = atol(line);
        i++;
        j = 0;
      }
    }
    else {
      goto escape;
    }
  }
  if (dimensions[2] > 255){
    goto escape;
  }
  /*after the last token, there should be a single whitespace character, which
   should have been consumed by the loop.
  */
  uint width = dimensions[0];
  uint height = dimensions[1];
  size_t length = width * height;
  void * mem = malloc(sizeof(Image) + length);
  Image *im = mem;
  im->width = width;
  im->height = height;
  im->data = mem + sizeof(*im);
  size_t read = fread(im->data, 1, length, f);
  if (read != length)
    goto late_escape;
  fclose(f);
  return im;

 late_escape:
  free(mem);
 escape:
  fclose(f);
 early_escape:
  return NULL;
}
