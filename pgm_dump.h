/* Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL/MPL2 */
#ifndef HAVE_PGM_DUMP
#define HAVE_PGM_DUMP

#define IMAGE_DIR "images/"

#include <string.h>
#include <math.h>
#include "recur-common.h"


/*pgm for greyscale */
static inline void
pgm_dump(const u8 *data, u32 width, u32 height, const char *name)
{
  FILE *fh = fopen(name, "w");
  if (fh == NULL){
    DEBUG("could not open '%s' for writing", name);
    return;
  }
  size_t size = width * height;
  fprintf(fh, "P5\n%u %u\n255\n", width, height);
  size_t wrote = fwrite(data, 1, size, fh);
  if (wrote != size){
    fprintf(stderr, "wanted to write %zu bytes; fwrite said %zu\n", size, wrote);
  }
  fflush(fh);
  fclose(fh);
}

static inline void
pgm_dump_auto_name(const u8 *data, u32 width, u32 height,
                   int stream_no, s64 stream_time, const char *id)
{
  static const char PGM_NAME_TEMPLATE[] = IMAGE_DIR "%s-%02d-%013lld-%ux%u.pgm";
  if (id == NULL)
    id = "img";
  char id_truncated[10];
  snprintf(id_truncated, sizeof(id_truncated), "%s", id);
  char pgm_name[sizeof(PGM_NAME_TEMPLATE) + 60];
  snprintf(pgm_name, sizeof(pgm_name), PGM_NAME_TEMPLATE,
           id_truncated,
           stream_no,
           (long long) stream_time,
           width, height);
  pgm_dump(data,
           width,
           height,
           pgm_name);
}

static inline void
pgm_dump_normalised_float(const float *data, u32 width, u32 height, const char *name)
{
  u8 bytes [width * height];
  for (u32 i = 0; i < width * height; i++){
    int b = data[i] * 255.99f;
    if (b < 0)
      b = 0;
    else if (b > 255)
      b = 255;
    bytes[i] = b;
  }
  pgm_dump(bytes, width, height, name);
}


static inline void
pgm_dump_unnormalised_float(const float *weights, int width, int height, const char *name)
{
  float biggest = 1e-35f;
  for (int i = 0; i < width * height; i++){
    float f = fabsf(weights[i]);
    if (f > biggest)
      biggest = f;
  }
  float scale = 255.99f / biggest;
  u8 *im = malloc_aligned_or_die(width * height);
  for (int i = 0; i < width * height; i++){
    im[i] = (u8)(fabs(weights[i]) * scale);
  }
  pgm_dump(im, width, height, name);
  MAYBE_DEBUG("%s biggest is %.2g", name, biggest);
  free(im);
}


/*pbm for bitmap */
static inline void
pbm_dump(const u64 *data64,
    const int width,
    const int height,
    const char *name)
{
  u8 *data = (u8*)data64;
  FILE *fh = fopen(name, "w");
  if (fh == NULL){
    DEBUG("could not open '%s' for writing", name);
    return;
  }
  fprintf(fh, "P4\n%u %u\n", width, height);
  int rx = width;
  u8 byte = 0;
  int offset = 0;
  for (int i = 0; i < width * height;){
    byte = data[i / 8];
    byte <<= offset;
    byte |= data[i / 8 + 1] >> (8 - offset);
    putc(~byte, fh);
    if (rx < 8){
      offset = (offset + rx) & 7;
      i += rx;
      rx = width;
    }
    else {
      i += 8;
      rx -= 8;
    }
  }

  fflush(fh);
  fclose(fh);
}


static inline void
putc_colourcoded_float(float f, FILE *fh){
  u8 b = fabsf(f);
  if (f < 0.0){
    putc(b, fh);
    putc(0, fh);
    putc(0, fh);
  }
  else if (f > 0.0){
    putc(0, fh);
    putc(b, fh);
    putc(0, fh);
  }
  else {/* zero is blue */
    putc(0, fh);
    putc(0, fh);
    putc(180, fh);
  }
}


static inline void
ppm_dump_signed_unnormalised_float(const float *weights, int width, int height, const char *name)
{
  float biggest = 1e-35f;
  for (int i = 0; i < width * height; i++){
    float f = fabsf(weights[i]);
    if (f > biggest)
      biggest = f;
  }
  float scale = 255.99f / biggest;
  MAYBE_DEBUG("%s biggest is %.2g", name, biggest);
  FILE *fh = fopen(name, "w");
  if (fh == NULL){
    DEBUG("could not open '%s' for writing", name);
    return;
  }
  fprintf(fh, "P6\n%u %u\n255\n", width, height);

  for (int i = 0; i < width * height; i++){
    putc_colourcoded_float(weights[i] * scale, fh);
  }
  fflush(fh);
  fclose(fh);
}


static inline void
ppm_dump_signed_unnormalised_colweighted_float(const float *weights, int width, int height, const char *name)
{
  float scales[width];
  memset(scales, 0, sizeof(scales));
  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      float f = fabsf(weights[y * width + x]);
      if (f > scales[x])
        scales[x] = f;
    }
  }
  for (int i = 0; i < width; i++){
    scales[i] = 255.99f / scales[i];
  }

  FILE *fh = fopen(name, "w");
  if (fh == NULL){
    DEBUG("could not open '%s' for writing", name);
    return;
  }
  fprintf(fh, "P6\n%u %u\n255\n", width, height);

  for (int y = 0; y < height; y++){
    for (int x = 0; x < width; x++){
      float f = weights[y * width + x] * scales[x];
      putc_colourcoded_float(f, fh);
    }
  }
  fflush(fh);
  fclose(fh);
}

static inline void
dump_weights_autoname(const float *weights, int width, int height,
    const char *desc, int id){
  char name[100];
  snprintf(name, sizeof(name), IMAGE_DIR "%s-%08d-%dx%d.pgm", desc, id, width, height);
  pgm_dump_unnormalised_float(weights, width, height, name);
}

static inline void
dump_colour_weights_autoname(const float *weights, int width, int height,
    const char *desc, int id){
  char name[100];
  snprintf(name, sizeof(name), IMAGE_DIR "%s-%08d-%dx%d.ppm", desc, id, width, height);
  ppm_dump_signed_unnormalised_float(weights, width, height, name);
}

enum {
  PGM_DUMP_GREY,
  PGM_DUMP_COLOUR,
};

typedef struct _TemporalPPM {
  float *im;
  int width;
  int height;
  int y;
  int id;
  char *basename;
  int counter;
  int mode;
  float **source;
} TemporalPPM;

static inline TemporalPPM *
temporal_ppm_alloc(int width, int height, const char *basename, int id,
    int mode, float **source){
  TemporalPPM *p = malloc(sizeof(TemporalPPM));
  p->im = malloc_aligned_or_die(width * height * sizeof(float));
  p->width = width;
  p->height = height;
  p->y = 0;
  p->id = id;
  p->counter = 0;
  p->basename = strdup(basename);
  p->mode = mode;
  p->source = source;
  return p;
}

static inline void
temporal_ppm_free(TemporalPPM *ppm){
  free(ppm->im);
  free(ppm->basename);
  free(ppm);
}

static inline void
temporal_ppm_write(TemporalPPM *ppm){
  char name[200];
  snprintf(name, sizeof(name), IMAGE_DIR "%s-%d-%08d-%dx%d.ppm",
      ppm->basename, ppm->id, ppm->counter, ppm->width, ppm->height);
  if (ppm->mode == PGM_DUMP_GREY){
    pgm_dump_unnormalised_float(ppm->im, ppm->width, ppm->height, name);
  }
  else {
    ppm_dump_signed_unnormalised_float(ppm->im, ppm->width, ppm->height, name);
  }
  ppm->y = 0;
  ppm->counter += ppm->height;
}

static inline void
temporal_ppm_add_row(TemporalPPM *ppm, const float *row){
  if (ppm == NULL){
    DEBUG("temporal PPM not initialised!");
    return;
  }
  memcpy(ppm->im + ppm->y * ppm->width, row, ppm->width * sizeof(float));
  ppm->y++;
  if (ppm->y == ppm->height){
    temporal_ppm_write(ppm);
  }
}

static inline void
temporal_ppm_row_from_source(TemporalPPM *ppm){
  if (ppm == NULL){
    DEBUG("temporal PPM not initialised!");
    return;
  }
  memcpy(ppm->im + ppm->y * ppm->width, *ppm->source, ppm->width * sizeof(float));
  ppm->y++;
  if (ppm->y == ppm->height){
    temporal_ppm_write(ppm);
  }
}

#endif
