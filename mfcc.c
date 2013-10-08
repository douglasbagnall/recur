#include "mfcc.h"
#include "pgm_dump.h"
#include "badmaths.h"

#define HZ_TO_MEL(x) (1127.0f * logf(1.0f + (x) / 700.0f))
#define MEL_TO_HZ(x) (700.0f * (expf((x) / 1127.0f) - 1.0f))

#define POWER(x) (x.r * x.r + x.i * x.i)

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
    slope = &ab->slopes[i];
    j = slope->left;
    /*left fractional part*/
    mul = slope->slope * slope->left_fraction;
    power = POWER(f[j]) * slope->left_fraction;

    sum_right = sum_left + (1.0f - mul) * power;
    /*Note sum_right is old sum_left */
    sum_left = mul * power;

    if (slope->left != slope->right){
      /*centre */
      for (j = slope->left + 1; j < slope->right; j++){
        mul += slope->slope;
        power = POWER(f[j]);
        sum_left += mul * power;
        sum_right += (1.0f - mul) * power;
      }
    }
    /*right fraction */
    mul += slope->slope * slope->right_fraction;
    power = POWER(f[j]) * slope->right_fraction;
    sum_left += mul * power;
    sum_right += (1.0f - mul) * power;

    if (i){
      ab->fft_bins[i - 1] = logf(sum_right + 0.01f) - slope->log_scale;
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

  recur_dct_cached(fft_bins, ab->dct_bins, ab->n_bins);
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
  float right = MEL_TO_HZ(mel) * hz_to_samples;
  for (i = 0; i < n_slopes; i++){
    RecurAudioBinSlope *s = &slopes[i];
    float left = right;
    s->left = (int)left;
    s->left_fraction = 1.0 - (left - s->left);
    mel += step;
    right = MEL_TO_HZ(mel) * hz_to_samples;
    s->right = (int)right;
    s->right_fraction = right - s->right;
    s->slope = 1.0 / (right - left);
    if (s->left == s->right){
      /*triangle is too little! */
      s->left_fraction = (right - left);
      s->right_fraction = 0;
    }
    s->log_scale = logf(1.0f + right - left);
  }
  return slopes;
}


static void
mfcc_slopes_dump2(RecurAudioBinner *ab){
  int i;
  int wsize = ab->window_size / ab->value_size;
  TemporalPPM *ppm = temporal_ppm_alloc(ab->n_bins, wsize, "mfcc", 0, PGM_DUMP_GREY);
  GstFFTF32Complex *f = calloc(sizeof(float), ab->window_size);
  for (i = 0; i <= wsize; i++){
    f[i].r = 1.0f;
    f[i].i = 1.0f;
    float *row = recur_bin_complex(ab, f);
    for (int j = 0; j < ab->n_bins; j++){
      row[j] = fast_expf(row[j]);
    }
    temporal_ppm_add_row(ppm, row);
    f[i].r = 0.0f;
    f[i].i = 0.0f;
  }
  temporal_ppm_free(ppm);
}

/*mfcc_slopes_dump draws a PGM showing whether the slope calculations worked*/
static void
mfcc_slopes_dump(RecurAudioBinner *ab){
  int i, j;
  int wsize = ab->window_size / ab->value_size;
  u8 *img = malloc_aligned_or_die(ab->n_bins * wsize);
  memset(img, 0, ab->n_bins * wsize);

  float mul;
  RecurAudioBinSlope *slope;
  /*first slope is left side only, last slope is right only*/
  for (i = 0; i <= ab->n_bins; i++){
    u8 *left = (i < ab->n_bins) ? img + i * wsize : NULL;
    u8 *right = (i) ? img + (i - 1) * wsize : NULL;
    slope = &ab->slopes[i];

    float sum_left = 0.0;
    float sum_right = 0.0;

    /*left fractional part*/
    mul = slope->slope * slope->left_fraction;
    if (left){
      left[slope->left] += 255 * mul * slope->left_fraction;
      sum_left += mul * slope->left_fraction;
    }
    if (right){
      right[slope->left] += 255 * (1.0 - mul) * slope->left_fraction;
      sum_right += (1.0 - mul) * slope->left_fraction;
    }
    if (slope->left != slope->right){
      /*centre */
      for (j = slope->left + 1; j < slope->right; j++){
        mul += slope->slope;
        if (left){
          left[j] += mul * 255;
          sum_left += mul;
        }
        if (right){
          right[j] += (1.0 - mul) * 255;
          sum_right += (1.0 - mul);
        }
      }
    }
    /*right fraction */
    mul += slope->slope * slope->right_fraction;
    if (left){
      left[slope->right] += 255 * mul * slope->right_fraction;
      sum_left += mul * slope->right_fraction;
    }
    if (right){
      right[slope->right] += 255 * (1.0f - mul) * slope->right_fraction;
      sum_right += (1.0f - mul) * slope->right_fraction;
    }

    MAYBE_DEBUG("%2d. left%3d right%3d slope %.3f fractions: L %.3f R %.3f  mul at end %.3f"
        " sum_L %.3f sum_R %.3f sum %.3f",
        i, slope->left, slope->right, slope->slope, slope->left_fraction,
        slope->right_fraction,  mul, sum_left, sum_right, sum_left + sum_right);
  }
  pgm_dump(img, wsize, ab->n_bins, IMAGE_DIR "/mfcc-bins.pgm");
  free(img);
}


void
recur_window_init(float *mask, int len, int type, float scale){
  int i;
  const double half_pi = G_PI * 0.5;
  const double pi_norm = G_PI / len;
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
      double z = pi_norm * (i + 0.5);
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
  mfcc_slopes_dump2(ab);
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


/* dct/idct based on recur/test/pydct.py, originally from Mimetic TV*/
/* XXX see ffmpeg's dct32.c for a relatively optimal dct32 (LGPL) */

void
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

void
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

/*recur_dct_cached is still a quadratic DCT, but it exploits the fact that the
  cos() calls are all using a small set of arguments, and caches the results
  in memory. Presumably because it can use double precision to calculate the
  cache, the results seem to be more accurate as well as faster.

  It uses a simple caching strategy: the calculated cosines are saved between
  consecutive transforms (the overwhelmingly common case so far) -- if the
  size of the transforms changes frequently the cache will be repeatedly
  recalculated, but this is still faster than the naive method.

  A proper DCT with butterflies and so on will be quicker, but needs to be
  tuned for its particular size.

  The indexing strategy has been determined experimentally -- other sequences
  may well be simpler.

  XXX a passed in cache pointer would be the thing if ever two DCT sizes are
  being used alternately.
 */

void
recur_dct_cached(const float *restrict input, float *restrict output, int len){
  int j, k;
  static float *cos_lut = NULL;
  static int cos_len = 0;
  if (cos_len != len * 2){
    if (cos_lut){
      free(cos_lut);
    }
    cos_len = len * 2;
    cos_lut = malloc_aligned_or_die((cos_len + 1) * sizeof(float));
    for (j = 0; j <= cos_len; j++){
      cos_lut[j] = cos(G_PI / cos_len * j);
    }
  }
  for (j = 0; j < len; j++){
    float a = 0.0f;
    int step = j * 2;
    int i = j;
    for (k = 0; k < len; k++){
      a += input[k] * cos_lut[i];
      i += step;
      if (i > cos_len){ /*bounce off the top */
        i = 2 * cos_len - i;
        step = -step;
      }
      else if (i < 0){
        i = -i;
        step = -step;
      }
    }
    output[j] = a;
  }
  output[0] *= 0.7071067811865476f;
}
