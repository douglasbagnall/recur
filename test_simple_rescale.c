#include "rescale.h"
#include "pgm_dump.h"
#include <math.h>

#define IMG_DIR "/home/douglas/recur/test-images"

static const char *names[] = {
  "cake",
  "zoo-odd-size",
  "butterfly"
};

const int RECUR_INPUT_WIDTH = 300;
const int RECUR_INPUT_HEIGHT = 200;


static void
save_scaled_image(u8 *im, int stride, int x, int y, int scale, const char *name){
  float *dest = calloc(1, RECUR_INPUT_WIDTH * RECUR_INPUT_HEIGHT * sizeof(float));

  recur_integer_downscale_to_float(im, dest, stride, x, y,
      RECUR_INPUT_WIDTH, RECUR_INPUT_HEIGHT, scale);

  pgm_dump_normalised_float(dest, RECUR_INPUT_WIDTH, RECUR_INPUT_HEIGHT, name);
  free(dest);
}

int
main(void){
  uint i, j;
  char name[100];
  for (j = 0; j < sizeof(names) / sizeof(names[0]); j++){
    snprintf(name, sizeof(name), "%s/%s.pgm", IMG_DIR, names[j]);
    Image * im = recur_load_pgm_file(name);
    for (i = 1; i < 6; i++){
      snprintf(name, sizeof(name), "%s/generated/simple-%s-%d.pgm", IMG_DIR, names[j], i);
      save_scaled_image(im->data, im->width, 100, 100, i, name);
    }
  }
}
