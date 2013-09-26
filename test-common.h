#include "recur-nn.h"
#include "pgm_dump.h"
#include <math.h>
#include "path.h"
#define SRC_TEXT TEST_DATA_DIR "/erewhon.txt"
#define SRC_TEXT2 TEST_DATA_DIR "/erewhon-erewhon"\
  "-revisited-sans-gutenberg.txt"


static inline int
search_for_max(float *answer, int len){
  ASSUME_ALIGNED(answer);
  int j;
  int best_offset = 0;
  float best_score = *answer;
  for (j = 1; j < len; j++){
    if (answer[j] >= best_score){
      best_score = answer[j];
      best_offset = j;
    }
  }
  return best_offset;
}


typedef struct _FloatIm {
  float * data;
  int width;
  int height;
  int next_row;
} FloatIm;

static inline void
init_error_image(FloatIm *im, int width, int height){
  im->width = width;
  im->height = height;
  im->data = malloc_aligned_or_die(width * height * sizeof(float));
  im->next_row = 0;
}

static inline void
add_error_image_row(FloatIm *im, float *o_error){
  if (im->next_row >= im->height){
    DEBUG("trying to add row %d/%d", im->next_row, im->height);
  }
  else {
    memcpy(im->data + im->next_row * im->width, o_error, im->width * sizeof(float));
  }
  im->next_row++;
}

static inline void
finish_error_image(FloatIm *im, char *name, int id){
  dump_colour_weights_autoname(im->data, im->width, im->height, name, id);
  free(im->data);
}
