#include "rescale.h"
#include "pgm_dump.h"
#include <math.h>
#include "path.h"

#define IMG_DIR TEST_DATA_DIR

static const int sizes[] = {
    512, 384,
    399, 399,
    1024, 768,
    999, 750,
    64, 48,
    30, 20
  };

static const char *names[] = {
  "zoo-odd-size",
  "butterfly"
};


void
compare_to_truth(const u8 *im, const int w, const int h,
    const char *im_name, const char *truth_name, const char *method_name){
  char name[100];
  snprintf(name, sizeof(name), "%s/generated/%s-%dx%d-%s.pgm",
      IMG_DIR, im_name, w, h, truth_name);

  Image * im2 = recur_load_pgm_file(name);
  if (im2){ /*in most cases, the file will be missing */
    double delta = 0;
    double delta2 = 0;
    for(int k = 0; k < w * h; k++){
      int d = im[k] - im2->data[k];
      delta2 += d * d;
      delta += abs(d);
    }
    printf("%11s vs %5s delta %.2f rms %.2f\n",
        truth_name, method_name,
        delta / (w * h),  sqrt(delta2 / (w * h)));
    free(im2);
  }
}


static void
report_delta(u8 *mem, u8 *mem2, int w, int h,
    const char *name, const char *msg){
  double delta2 = 0.0;
  double delta = 0.0;
  for(int k = 0; k < w * h; k++){
    int d = mem[k] - mem2[k];
    delta2 += d * d;
    delta += abs(d);
  }
  printf("%12s %4dx%-4d %30s %.2f rms %.2f\n",
      name, w, h, msg, delta / (w * h), sqrt(delta2 / (w * h)));
}

int
main(void){
  uint i, j;
  char name[100];
  for (j = 0; j < sizeof(names) / sizeof(names[0]); j++){
    snprintf(name, sizeof(name), "%s/%s.pgm", IMG_DIR, names[j]);
    Image * im = recur_load_pgm_file(name);
    snprintf(name, sizeof(name), "/tmp/%s-dup.pgm", names[j]);
    pgm_dump(im->data, im->width, im->height, name);
    for (i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i += 2){
      int w = sizes[i];
      int h = sizes[i + 1];
      u8 *mem = malloc(w * h);

      snprintf(name, sizeof(name), IMG_DIR "/output/%s-%dx%d.pgm", names[j], w, h);
      recur_adaptive_downscale(im->data, im->width, im->height, im->width,
          mem, w, h, w);
      pgm_dump(mem, w, h, name);

      snprintf(name, sizeof(name), IMG_DIR "/output/%s-%dx%d-exact.pgm",
          names[j], w, h);
      START_TIMER(exact);
      recur_exact_downscale(im->data, im->width, im->height, im->width,
          mem, w, h, w);
      DEBUG_TIMER(exact);
      pgm_dump(mem, w, h, name);

      u8* mem2 = malloc(w * h);
      snprintf(name, sizeof(name), IMG_DIR "/output/%s-%dx%d-skipping.pgm",
          names[j], w, h);
      START_TIMER(skipping);
      recur_skipping_downscale(im->data, im->width, im->height, im->width,
          mem2, w, h, w);
      DEBUG_TIMER(skipping);

      pgm_dump(mem2, w, h, name);

      /*float -- convert to and from u8 */
      float* float_mem = malloc(w * h * sizeof(float));
      u8* mem3 = malloc(w * h);
      float* float_im = malloc(im->width * im->height * sizeof(float));
      for (uint k = 0; k < im->width * im->height; k++){
        float_im[k] = im->data[k];
      }
      snprintf(name, sizeof(name), IMG_DIR "/output/%s-%dx%d-float.pgm",
          names[j], w, h);
      START_TIMER(float_);
      recur_float_downscale(float_im, im->width, im->height, im->width,
          float_mem, w, h, w);
      DEBUG_TIMER(float_);
      for (int k = 0; k < w * h; k++){
        mem3[k] = float_mem[k];
      }
      pgm_dump(mem3, w, h, name);
      free(float_mem);
      free(float_im);



      report_delta(mem, mem2, w, h, names[j], "skipping vs exact delta");
      report_delta(mem, mem3, w, h, names[j], "float vs exact delta");

      compare_to_truth(mem, w, h, names[j], "Magick", "exact");
      compare_to_truth(mem, w, h, names[j], "PIL", "exact");
      compare_to_truth(mem3, w, h, names[j], "Magick", "float");
      compare_to_truth(mem3, w, h, names[j], "PIL", "float");

      free(mem3);
      free(mem2);
      free(mem);
    }
    free(im);
  }
}
