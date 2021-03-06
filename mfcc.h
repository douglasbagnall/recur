/* Copyright 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL */
#ifndef HAVE_MFCC_H
#define HAVE_MFCC_H

#include "recur-common.h"
#include <gst/fft/gstfftf32.h>


typedef struct _RecurAudioBinSlope RecurAudioBinSlope;

struct _RecurAudioBinSlope {
  int left;
  int right;
  float left_fraction;
  float right_fraction;
  float log_scale;
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
recur_extract_log_freq_bins(RecurAudioBinner *ab, float *data);

float *
recur_extract_mfccs(RecurAudioBinner *ab, float *data);

RecurAudioBinSlope *
recur_bin_slopes_new(const int n_bins, const int fft_len,
    const float fmin, const float fmax, const float fknee,
    const float ffocus, const float audio_rate) __attribute__ ((malloc));


RecurAudioBinner *
recur_audio_binner_new(int window_size, int window_type,
    int n_bins,
    float min_freq,
    float max_freq,
    float knee_freq,
    float focus_freq,
    float audio_rate,
    float scale,
    int value_size
);

void recur_audio_binner_delete(RecurAudioBinner *ab);

void recur_window_init(float *mask, int len, int type, float scale);


void recur_dct(const float *restrict input, float *restrict output, int len);
void recur_dct_cached(const float *restrict input, float *restrict output, int len);
void recur_idct(const float *restrict input, float *restrict output, int len);
#endif
