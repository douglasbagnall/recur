#include "recur-common.h"
#include "mfcc.h"

#define N_FFT_BINS 32
#define MIN_FREQ 0
#define MAX_FREQ (AUDIO_RATE * 0.499)
#define KNEE_FREQ 700
#define FOCUS_FREQ 600
#define AUDIO_RATE 8000
#define WINDOW_SIZE 512

int
main(void){
  /*This draws the images of the audio bins */
  recur_audio_binner_new(WINDOW_SIZE, RECUR_WINDOW_HANN,
      N_FFT_BINS, MIN_FREQ, MAX_FREQ, KNEE_FREQ, FOCUS_FREQ,
      AUDIO_RATE, 1.0, 2);
}
