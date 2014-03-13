#include "recur-common.h"
#include "mfcc.h"
#include "pgm_dump.h"
#include <gst/fft/gstfftf32.h>
#include "path.h"

#define RECUR_N_FFT_BINS 40
#define RECUR_MFCC_MIN_FREQ 20
#define RECUR_MFCC_MAX_FREQ (RECUR_AUDIO_RATE * 0.499)
#define RECUR_MFCC_KNEE_FREQ 700
#define RECUR_AUDIO_RATE 16000


int
main(void){
  int fps_d = 1;
  int fps_n = 25;
  int expected_samples = RECUR_AUDIO_RATE * fps_d / fps_n;
  int min_window_size = ROUND_UP_4(expected_samples * 3 / 2);
  int window_size = gst_fft_next_fast_length (min_window_size);

  RecurAudioBinSlope * bins = recur_bin_slopes_new(RECUR_N_FFT_BINS,
      window_size / 2,
      RECUR_MFCC_MIN_FREQ,
      RECUR_MFCC_MAX_FREQ,
      700, 0,
      RECUR_AUDIO_RATE
  );

  printf("window size %d bins %d\n", window_size,
      RECUR_N_FFT_BINS);

  u8 *img = malloc_aligned_or_die(RECUR_N_FFT_BINS * window_size / 2);

  int i, j;
  float mul = 0.0;
  RecurAudioBinSlope *bin = &bins[0];
  for (j = bin->left; j < bin->right; j++){
    mul += bin->slope;
    img[j] = mul * 255;
  }
  printf("%2d. left %3d right %3d slope %f mul at end %f\n",
      0, bin->left, bin->right, bin->slope, mul);

  for (i = 1; i < RECUR_N_FFT_BINS; i++){
    bin = &bins[i];
    mul = 0.0;
    for (j = bin->left; j < bin->right; j++){
      mul += bin->slope;
      img[i * window_size / 2 + j] = mul * 255;
      img[(i - 1) * window_size / 2 + j] = (1.0 - mul) * 255;
    }
    printf("%2d. left %3d right %3d slope %f mul at end %f\n",
        i, bin->left, bin->right, bin->slope, mul);
  }
  bin = &bins[i];
  mul = 0.0;
  for (j = bin->left; j < bin->right; j++){
    mul += bin->slope;
    //img[i * window_size / 2 + j] = mul * 255;
    img[(i - 1) * window_size / 2 + j] = (1.0 - mul) * 255;
  }
    printf("%2d. left %3d right %3d slope %f mul at end %f\n",
        i, bin->left, bin->right, bin->slope, mul);
  pgm_dump(img, window_size / 2, RECUR_N_FFT_BINS, DEBUG_IMAGE_DIR "/mfcc4.pgm");
}
