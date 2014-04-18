/* Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL/MPL2 */
#include "recur-common.h"
#include <math.h>
#include "badmaths.h"
#include <stdio.h>
#include <fenv.h>
#include <ctype.h>
#include "mdct.h"
#include "window.h"

//#define TEST_AUDIO_FILE "test-audio/494-slower-huge.wav"
#define TEST_AUDIO_FILE "test-audio/371-slower-huge.wav"
#define DEST_AUDIO_FILE "mdct-out.wav"

#define WINDOW_BITS 9
#define WINDOW_SIZE (1 << WINDOW_BITS)
#define WINDOW_NO (WINDOW_BITS - 6)


static inline void
apply_window(float *restrict samples, const float *restrict window){
  int half_window = WINDOW_SIZE / 2;
  int i;
  for (i = 0; i < half_window; i++){
    samples[i] *= window[i];
  }
  for (i = 0; i < half_window; i++){
    samples[half_window + i] *= window[half_window - 1 - i];
  }
}


int
main(void){
  int i;
  const int half_window = WINDOW_SIZE / 2;
  s16 buffer[half_window];
  s16 buffer2[half_window];
  float *pcm1 = calloc(WINDOW_SIZE, sizeof(float));
  float *pcm2 = calloc(WINDOW_SIZE, sizeof(float));
  //float *pcm3 = calloc(WINDOW_SIZE, sizeof(float));

  mdct_lookup m_look;
  mdct_init(&m_look, WINDOW_SIZE);
  const float *window = _vorbis_window_get(WINDOW_NO);

  FILE *inf = fopen_or_abort(TEST_AUDIO_FILE, "r");
  FILE *outf = fopen_or_abort(DEST_AUDIO_FILE, "w");

  /*wav header is 44 bytes. reuse it as-is.*/
  int len = fread(buffer, 1, 44, inf);
  fwrite(buffer, 1, 44, outf);

  for (;;){
    len = fread(buffer, sizeof(s16), half_window, inf);
    DEBUG("read %d bytes", len);
    if (len == 0)
      break;
    if(len < half_window){
      memset(buffer + len, 0,
          (half_window - len) * sizeof(s16));
    }
    for(i = 0; i < half_window; i++){
      float s = buffer[i] / 32768.0f;
      pcm1[i] = s * window[i];
      pcm2[half_window + i] = s * window[half_window - 1 - i];
    }
    mdct_forward(&m_look, pcm2, pcm2);
    mdct_backward(&m_look, pcm2, pcm2);

    for(i = 0; i < half_window; i++){
      float s = (pcm1[half_window + i] * window[half_window - 1 - i] +
          pcm2[i] * window[i]);
      //float s = pcm1[half_window + i] + pcm2[i];
      buffer2[i] = s * 32767.0f;
    }
    len = fwrite(buffer2, sizeof(s16), half_window, outf);
    DEBUG("wrote %d bytes buffer[0] %d buffer2[0] %d", len, buffer[0], buffer2[0]);
    float *tmp = pcm1;
    pcm1 = pcm2;
    pcm2 = tmp;
  }
  free(pcm1);
  free(pcm2);
}
