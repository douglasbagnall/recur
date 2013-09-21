#include "mfcc.h"
#include "pgm_dump.h"

#define HZ_TO_MEL(x) (1127.0f * logf(1.0f + (x) / 700.0f))
#define MEL_TO_HZ(x) (700.0f * (expf((x) / 1127.0f) - 1.0f))

/*recur_bin_real puts already calculated real-valued frequency data (e.g.
  mdct) into log squared mel bins */
float *
recur_bin_real(RecurAudioBinner *ab, float *data)
{
  int i, j;
  float mul;
  float sum_left = 0.0f;
  float sum_right;
  float power;
  RecurAudioBinSlope *slope;
  /*first slope is left side only, last slope is right only*/
  for (i = 0; i <= ab->n_bins; i++){
    sum_right = sum_left;
    sum_left = 0.0f;
    mul = 0.0f;
    slope = &ab->slopes[i];
    for (j = slope->left; j < slope->right; j++){
      power = data[j] * data[j];
      sum_left += mul * power;
      sum_right += (1.0f - mul) * power;
      mul += slope->slope;
    }
    if (i){
      ab->fft_bins[i - 1] = logf(sum_right + 0.01f);
    }
  }
  return ab->fft_bins;
}

float *
recur_bin_complex(RecurAudioBinner *ab, GstFFTF32Complex *f)
{
  int i, j;
  float mul;
  float sum_left = 0.0f;
  float sum_right;
  float power;
  RecurAudioBinSlope *slope;
  /*first slope is left side only, last slope is right only*/
  for (i = 0; i <= ab->n_bins; i++){
    sum_right = sum_left;
    sum_left = 0.0f;
    mul = 0.0f;
    slope = &ab->slopes[i];
    for (j = slope->left; j < slope->right; j++){
      power = f[j].r * f[j].r + f[j].i * f[j].i;
      //DEBUG("%d: %g + %gi = %g", j, f[j].r, f[j].i, power);
      sum_left += mul * power;
      sum_right += (1.0f - mul) * power;
      mul += slope->slope;
    }
    /*scale for approximately equal triangles */
    //sum_right *= slope->slope;
    if (i){
      ab->fft_bins[i - 1] = logf(sum_right + 0.01f);
      //DEBUG("%d: %f", i - 1, ab->fft_bins[i - 1]);
    }
  }
  return ab->fft_bins;
}


/* Apply the cached window function, returning the actually used destination.

*/

const float *
recur_apply_window(RecurAudioBinner *ab, const float *src, float *dest)
{
  int i;
  if (ab->window_type == RECUR_WINDOW_NONE){
    if (dest != src){
      memmove(dest, src, ab->window_size * sizeof(float));
    }
    else {
      return src;
    }
  }
  else {
    for (i = 0; i < ab->window_size; i++){
      dest[i] = src[i] * ab->mask[i];
    }
  }
  return dest;
}

/* extract log scaled, mel-shaped, frequency bins */
float *
recur_extract_log_freq_bins(RecurAudioBinner *ab, float *data){
  /* XXX assumes ab->value_size is 2 */
  const float *windowed_data = recur_apply_window(ab, data, data);
  gst_fft_f32_fft(ab->fft,
      windowed_data,
      ab->freq_data
  );
  return recur_bin_complex(ab, ab->freq_data);
}

float *
recur_extract_mfccs(RecurAudioBinner *ab, float *data){
  float *fft_bins = recur_extract_log_freq_bins(ab, data);

  recur_dct(fft_bins, ab->dct_bins, ab->n_bins);
  return ab->dct_bins;
}


RecurAudioBinSlope * __attribute__((malloc))
recur_bin_slopes_new(const int n_bins, const int fft_len,
                     const float fmin, const float fmax,
                     const float audio_rate){
  const int n_slopes = n_bins + 1;
  RecurAudioBinSlope * slopes = malloc_aligned_or_die(n_slopes *
                                                      sizeof(RecurAudioBinSlope));
  int i;
  float mmin = HZ_TO_MEL(fmin);
  float mmax = HZ_TO_MEL(fmax);
  float step = (mmax - mmin) / n_slopes;
  float mel = mmin;
  float hz_to_samples = fft_len * 2 / audio_rate;
  for (i = 0; i < n_slopes; i++){
    RecurAudioBinSlope *s = &slopes[i];
    s->left = (int)(MEL_TO_HZ(mel) * hz_to_samples + 0.5);
    do {
      step = (mmax - mel) / (n_slopes - i);
      mel += step;
      s->right = (int)(MEL_TO_HZ(mel) * hz_to_samples + 0.5);
    } while (s->right == s->left);
    s->slope = 1.0 / (s->right - s->left);
  }
  return slopes;
}

/*mfcc_slopes_dump draws a PGM showing whether the slope calculations worked*/
static void
mfcc_slopes_dump(RecurAudioBinner *ab){
  int i, j;
  int wsize = ab->window_size / ab->value_size;
  u8 *img = malloc_aligned_or_die(ab->n_bins * wsize);
  memset(img, 0, ab->n_bins * wsize);
  float mul = 0.0;
  RecurAudioBinSlope *bin;

  MAYBE_DEBUG("window size %d value size %d bins %d\n",
      ab->window_size, ab->value_size, ab->n_bins);

  for (i = 0; i < ab->n_bins; i++){
    bin = &ab->slopes[i];
    mul = 0.0;
    for (j = bin->left; j < bin->right; j++){
      mul += bin->slope;
      img[i * wsize + j] = mul * 255;
      if (i){
        img[(i - 1) * wsize + j] = (1.0 - mul) * 255;
      }
    }
    MAYBE_DEBUG("%2d. left %3d right %3d slope %f mul at end %f\n",
        i, bin->left, bin->right, bin->slope, mul);
  }
  bin = &ab->slopes[i];
  mul = 0.0;
  for (j = bin->left; j < bin->right; j++){
    mul += bin->slope;
    //img[i * size + j] = mul * 255;
    img[(i - 1) * wsize + j] = (1.0 - mul) * 255;
  }
  MAYBE_DEBUG("%2d. left %3d right %3d slope %f mul at end %f\n",
      i, bin->left, bin->right, bin->slope, mul);
  pgm_dump(img, wsize, ab->n_bins, IMAGE_DIR "/mfcc-bins.pgm");
  free(img);
}


void
recur_window_init(float *mask, int len, int type, float scale){
  int i;
  const float half_pi = G_PI * 0.5;
  //const float half_pi_norm = G_PI * 0.5 / len;
  const float pi_norm = G_PI / len;
  switch (type){
  case RECUR_WINDOW_HANN:
    for (i = 0; i < len; i++){
      mask[i] = (0.5 - 0.5 * cos(2.0 * pi_norm * i)) * scale;
    }
    break;
  case RECUR_WINDOW_MP3:
    for (i = 0; i < len; i++){
      mask[i] = sin(pi_norm * (i + 0.5f)) * scale;
    }
    break;
  case RECUR_WINDOW_VORBIS:
    for (i = 0; i < len; i++){
      float z = pi_norm * (i + 0.5f);
      mask[i] = sin(half_pi * sin(z) * sin(z)) * scale;
    }
    break;
 case RECUR_WINDOW_NONE:
 default:
    for (i = 0; i < len; i++){
      mask[i] = 1.0f;
    }
    break;
  }
}


RecurAudioBinner *  __attribute__((malloc))
recur_audio_binner_new(int window_size, int window_type,
    int n_bins,
    float min_freq,
    float max_freq,
    float audio_rate,
    float scale,
    int value_size /*1 for real, 2 for complex*/
){
  RecurAudioBinner *ab = calloc(1, sizeof(*ab));
  ab->window_size = window_size;
  ab->window_type = window_type;
  ab->n_bins = n_bins;
  ab->pcm_data = malloc_aligned_or_die((window_size + 2) * sizeof(float));
  ab->freq_data = malloc_aligned_or_die((window_size + 2) * sizeof(float));
  ab->fft = gst_fft_f32_new(window_size, FALSE);

  float *mask = malloc_aligned_or_die((window_size + 2) * sizeof(float));
  recur_window_init(mask, window_size, window_type, scale);
  ab->mask = mask;

  ab->value_size = value_size;
  ab->slopes = recur_bin_slopes_new(n_bins,
      window_size / value_size,
      min_freq,
      max_freq,
      audio_rate
  );
  mfcc_slopes_dump(ab);

  ab->fft_bins = malloc_aligned_or_die((n_bins + 2) * sizeof(float));
  ab->dct_bins = malloc_aligned_or_die((n_bins + 2) * sizeof(float));
  return ab;
}

void
recur_audio_binner_delete(RecurAudioBinner *ab){
  free(ab->slopes);
  free(ab->pcm_data);
  free(ab->freq_data);
  free((void *)ab->mask);
  free(ab->fft_bins);
  free(ab->dct_bins);
  gst_fft_f32_free(ab->fft);
  free(ab);
}
