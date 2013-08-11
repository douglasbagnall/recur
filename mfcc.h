#include "recur-common.h"
#include <gst/fft/gstfftf32.h>


typedef struct _RecurAudioBinSlope RecurAudioBinSlope;

struct _RecurAudioBinSlope {
  int left;
  int right;
  float slope;
};

typedef struct _RecurAudioBinner {
  GstFFTF32 *fft;
  float *pcm_data;
  GstFFTF32Complex *freq_data;
  RecurAudioBinSlope *slopes;
  int window_size;
  int n_bins;
  const float *mask;
  float *fft_bins;
  float *dct_bins;
  int window_type;
  int value_size;
} RecurAudioBinner;

enum {
  RECUR_WINDOW_NONE = 0,
  RECUR_WINDOW_HANN = 1,
  RECUR_WINDOW_VORBIS,
  RECUR_WINDOW_MP3,
};

float *
recur_extract_log_freq_bins(RecurAudioBinner *ab, const float *data);

float *
recur_extract_mfccs(RecurAudioBinner *ab, float *data);

RecurAudioBinSlope *
recur_bin_slopes_new(const int n_bins, const int fft_len,
                     const float fmin, const float fmax,
                     const float audio_rate) __attribute__ ((malloc));


RecurAudioBinner *
recur_audio_binner_new(int window_size, int window_type,
    int n_bins,
    float min_freq,
    float max_freq,
    float audio_rate,
    int value_size
);

void recur_audio_binner_delete(RecurAudioBinner *ab);

//void recur_dct(const float *restrict input, float *restrict output, int len);
//void recur_idct(const float *restrict input, float *restrict output, int len);

/* dct/idct based on recur/test/pydct.py, originally from Mimetic TV*/

static inline void
recur_dct(const float *restrict input, float *restrict output, int len){
  int j, k;
  float pin = G_PI / len;
  for (j = 0; j < len; j++){
    float a = 0.0f;
    for (k = 0; k < len; k++){
      a += input[k] * cosf(pin * j * (k + 0.5f));
    }
    output[j] = a;
  }
  output[0] *= 0.7071067811865476f;
}

static inline void
recur_idct(const float *restrict input, float *restrict output, int len){
  int j, k;
  float pin = G_PI / len;
  float scale = 2.0f / len;
  for (j = 0; j < len; j++){
    float a = 0.7071067811865476f * input[0];
    for (k = 1; k < len; k++){
      a += input[k] * cosf(pin * k * (j + 0.5f));
    }
    output[j] = a * scale;
  }
}
