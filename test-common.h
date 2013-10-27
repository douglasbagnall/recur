#include "recur-nn.h"
#include "pgm_dump.h"
#include <math.h>
#include "path.h"
#define DICKENS_SHUFFLED_TEXT TEST_DATA_DIR "/dickens-shuffled.txt"
#define DICKENS_TEXT TEST_DATA_DIR "/dickens.txt"
#define EREWHON_TEXT TEST_DATA_DIR "/erewhon.txt"
#define EREWHON_LONG_TEXT TEST_DATA_DIR "/erewhon-erewhon"\
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
