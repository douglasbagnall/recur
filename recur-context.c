/* GStreamer
 * Copyright (C) <2013> Douglas Bagnall <douglas@halo.gen.nz>
 *
 */
#include "recur-context.h"
#include "badmaths.h"
#include "context-helpers.h"
#include "blit-helpers.h"
#include "recur-common.h"
#include "recur-nn.h"
#include "rescale.h"
#include <string.h>
#include <stdio.h>

GST_DEBUG_CATEGORY(recur_context_debug);
#define GST_CAT_DEFAULT recur_context_debug

#define SAMPLES_TO_NS(x) ((x) * BILLION / (RECUR_AUDIO_RATE * RECUR_AUDIO_CHANNELS))
#define NS_TO_SAMPLES(x) ((x) * RECUR_AUDIO_CHANNELS * RECUR_AUDIO_RATE / BILLION)

enum RecurAudioAnswer {
  BAD_VIDEO_PTS = -1,
  AUDIO_OK,
  NO_AUDIO
};


static enum RecurAudioAnswer
consume_audio_samples(GQueue *queue, float *destination,
                      GstClockTime centre_time, size_t window_size){
  GstClockTime duration = SAMPLES_TO_NS(window_size);
  if (duration / 2 > centre_time){
    GST_LOG("Too early: duration %llu, centre_time %llu",
        duration, centre_time);
    return BAD_VIDEO_PTS;
  }

  GstClockTime start_time = centre_time - duration / 2;
  GstClockTime end_time = start_time + duration;
  GstBuffer *b;
  GstClockTime a_start, a_end;
  /*The head of the queue is oldest -> lowest pts.
     we need:
     1) the buffer which contains the start_time (first a_start <= start_time)
     2) the buffer containing the end_time (last a_start >= end_time)
     3) any buffers in between
   */
  /* a_start > end:   newer than end, no use
     a_start >= start: newer than start, part of buffer
     a_start + duration < start: too old, discard.
   */
  GST_LOG("recur samples %zu, duration %llu, start %llu end %llu",
      window_size, duration, start_time, end_time);
  GST_LOG("recur video centre time %" GST_TIME_FORMAT " == %llu",
      GST_TIME_ARGS(centre_time), centre_time);

  for (;;){
    b = g_queue_peek_head(queue);
    if (b == NULL){
      GST_DEBUG("Audio queue is empty!");
      return NO_AUDIO;
    }
    a_start = GST_BUFFER_PTS(b);
    a_end = a_start + GST_BUFFER_DURATION(b);
    GST_LOG("recur audio start time %" GST_TIME_FORMAT " == %llu",
        GST_TIME_ARGS(a_start), a_start);
    GST_LOG("recur audio end time %" GST_TIME_FORMAT " == %llu",
        GST_TIME_ARGS(a_end), a_end);

    if (a_end < start_time){
      GST_LOG("buffer is too old, discarding: audio end %"
          GST_TIME_FORMAT "; < start %" GST_TIME_FORMAT,
          GST_TIME_ARGS(a_end), GST_TIME_ARGS(start_time));
      b = g_queue_pop_head(queue);
      gst_buffer_unref(b);
      continue;
    }
    if (a_start > start_time){
      /*This is the oldest audio buffer, so if the audio starts in the middle
        of this one, there won't be another buffer to fill in the beginning. */
      GST_LOG("audio buffer is too new: audio start %"
                GST_TIME_FORMAT " > video start %" GST_TIME_FORMAT,
          GST_TIME_ARGS(a_start), GST_TIME_ARGS(start_time));
      return NO_AUDIO;
    }
    /*so we have a buffer where
      a_start <= start_time <= a_end
      and here lets exit this loop, because the buffers might be needed by
      later frames, so popping them is wrong and a peek_nth loop is needed.
     */
    break;
  }

  int end = window_size - 1;
  GstMapInfo map;
  for (int i = 0;; i++){
    b = g_queue_peek_nth(queue, i);
    if (b == NULL){
      GST_LOG("audio queue is too short, missing item %d", i);
      return NO_AUDIO;
    }
    a_start = GST_BUFFER_PTS(b);
    a_end = a_start + GST_BUFFER_DURATION(b);
    int a_samples = NS_TO_SAMPLES(a_end - a_start);
    int a_samples2 = b->offset_end - b->offset;
    GST_LOG("audio samples from time %d;  from metadata %d", a_samples, a_samples2);
    gst_buffer_map(b, &map, 0);
    audio_sample *audio = (audio_sample *)map.data;
    int j = a_samples - 1;
    if (a_end > end_time){
      j -= NS_TO_SAMPLES(a_end - end_time);
    }
    for (; j >= 0 && end >= 0; j--, end--){
      destination[end] = audio[j];
    }
    if (end < 0)
      break;
  }
  return AUDIO_OK;
}

/* extract audio feature from pre-filled context->audio_window,
   applying window function in-place,
   putting fft results in context->audio_freq,
   normalising by DC coefficient
   and placing the result in context->current_audio
*/

static void
extract_audio_features(RecurContext *context){
  const float *dct_bins = recur_extract_mfccs(context->audio_binner,
      context->audio_binner->pcm_data);

  /*normalise all by dc component (volume), and replace dc component by change from
    previous volume */
  float previous_volume = context->audio_volume;
  context->audio_volume = dct_bins[0];
  float scale = 1.0f / (dct_bins[0] ? dct_bins[0] : 1.0f);
  context->current_audio[0] = (dct_bins[0] - previous_volume) * scale;
  for (int i = 1; i < RECUR_N_MFCCS; i++){
    context->current_audio[i] = dct_bins[i]  * scale;
  }
}

static void
extract_video_features(GstBuffer *buffer, RecurFrame *thumb, GstVideoInfo *video_info)
{
  GstVideoFrame frame;
  gst_video_frame_map (&frame, video_info, buffer, GST_MAP_READ);

  thumb->centre_time = GST_BUFFER_PTS(buffer) + GST_BUFFER_DURATION(buffer) / 2;

  int i;
  u8 *d = (u8*)&thumb->Y;
  for (i = 0; i < 3; i++){
    /*convert first to working size */
    const u8 *src = GST_VIDEO_FRAME_COMP_DATA(&frame, i);
    int sw = GST_VIDEO_FRAME_COMP_WIDTH(&frame, i);
    int sh = GST_VIDEO_FRAME_COMP_HEIGHT(&frame, i);
    int ss = GST_VIDEO_FRAME_COMP_STRIDE(&frame, i);
    recur_adaptive_downscale(src, sw, sh, ss,
        d, RECUR_WORKING_WIDTH, RECUR_WORKING_HEIGHT,
        RECUR_WORKING_WIDTH);
    d += RECUR_WORKING_WIDTH * RECUR_WORKING_HEIGHT;
  }
  gst_video_frame_unmap (&frame);
}


void
recur_queue_video_buffer(RecurContext *context, GstBuffer *buffer)
{
  RecurFrame *frame = &context->frame_queue[context->fq_tail];
  RECUR_FQ_ADVANCE(context->fq_tail);

  extract_video_features(buffer, frame, &context->video_info);
  GST_LOG("fq_head is %i, tail is %i", context->fq_head, context->fq_tail);
}

void
recur_fill_video_frame(RecurContext *context, GstVideoFrame *dest)
{
  dump_frame(dest);
  RecurFrame *src_frame = &context->frame_queue[context->fq_head];
  RecurFrame *target_frame = &context->frame_queue[RECUR_FQ_NEXT(context->fq_head)];
  RECUR_FQ_ADVANCE(context->fq_head);
  src_frame->pending = 0;
  GstClockTime centre_time = src_frame->centre_time;

  GST_LOG("recur centre time %" GST_TIME_FORMAT " == %llu",
      GST_TIME_ARGS(centre_time),
      centre_time);

  RecurAudioBinner *ab = context->audio_binner;
  enum RecurAudioAnswer audio_result = consume_audio_samples(&context->audio_queue,
      ab->pcm_data, centre_time, ab->window_size);
  if (audio_result == NO_AUDIO){
    GST_DEBUG("No audio!");
    goto grey;
  }
  else if (audio_result == BAD_VIDEO_PTS){
    GST_DEBUG("bad video PTS");
    /* XXX and what? */
  }

  extract_audio_features(context);
  if (RECUR_TRAIN)
    recur_train_nets(context, src_frame, target_frame);

  u8 *Y = context->planes;
  u8 *Cb = Y + RECUR_CONFAB_PLANE_SIZE;
  u8 *Cr = Cb + RECUR_CONFAB_PLANE_SIZE;
  recur_confabulate(context, Y, Cb, Cr);

  if (context->osdebug){
    blit_thumbnail(context, dest, 4, 4);
    blit_planar_float(dest, context->seed, 100, 5,
        RECUR_INPUT_WIDTH, RECUR_INPUT_HEIGHT, 2, false);
    GST_LOG("RECUR_CONFAB_PLANE_SIZE is %d (%dx%d)", RECUR_CONFAB_PLANE_SIZE,
        RECUR_CONSTRUCTOR_WIDTH, RECUR_CONSTRUCTOR_HEIGHT);

    blit_planar_u8(dest, Y, 10, 80, RECUR_CONSTRUCTOR_WIDTH, RECUR_CONSTRUCTOR_HEIGHT, 2);
    for (int i = 0; i < 8; i++){
      RecurNN *net = context->constructors[i];
      blit_planar_float(dest, net->real_inputs + RECUR_N_MFCCS, 110 + i * 30, 10,
          RECUR_INPUT_WIDTH, RECUR_INPUT_HEIGHT, 4, false);
      blit_planar_float(dest, net->output_layer, 110 + i * 30, 30, /*sigmoid blit */
          RECUR_OUTPUT_WIDTH, RECUR_OUTPUT_HEIGHT, 2, true);
    }
  }
  else {
    fill_from_planar_u8(dest, Y,
        RECUR_CONSTRUCTOR_WIDTH, RECUR_CONSTRUCTOR_HEIGHT);
  }
  possibly_save_state(context);
  return;

 grey:
  context->video_lag++;
  GST_LOG("sending grey. frame queue head is %u; tail is %d; lag is %d",
      context->fq_head, context->fq_tail, context->video_lag);
  for (uint i = 0; i < GST_VIDEO_INFO_N_PLANES(&context->video_info); i++){
    GstMapInfo *map = &dest->map[i];
    memset(map->data, 127, map->size);
  }
  return;
}


void
recur_queue_audio_segment(RecurContext *context, GstBuffer *buffer)
{
  gst_buffer_ref(buffer);
  GST_LOG("queueing audio starting %llu, ending %llu",
      GST_BUFFER_PTS(buffer), GST_BUFFER_PTS(buffer) + GST_BUFFER_DURATION(buffer));
  g_queue_push_tail(&context->audio_queue, buffer);
  GST_LOG("queue is now %u long", g_queue_get_length (&context->audio_queue ));

}

void
recur_fill_audio_segment(RecurContext *context, GstBuffer *buffer)
{
  GstMapInfo info;
  gst_buffer_map(buffer, &info, GST_MAP_WRITE);
  //s16 *samples = (s16 *) info->data;
  //uint size = info->size / sizeof(s16);
  //XXX gst_buffer_memset (buffer, 0, 0, -1);
  memset(info.data, 0, info.size);
  gst_buffer_unmap(buffer, &info);
}

void recur_context_init(RecurContext *context){
  GST_DEBUG_CATEGORY_INIT (recur_context_debug, "recur_context", RECUR_LOG_COLOUR,
      "recur context");
  memset(context, 0, sizeof(RecurContext));
  g_queue_init(&context->audio_queue);
  size_t fq_mem = RECUR_FQ_LENGTH * sizeof(RecurFrame);
  context->frame_queue = malloc_aligned_or_die(fq_mem);
  memset(context->frame_queue, 0, fq_mem);
  recur_setup_nets(context, NET_LOG_FILE);
  context->planes = malloc_aligned_or_die(RECUR_CONFAB_PLANE_SIZE * 3);
}

void recur_context_finalize(RecurContext *context){
  GstBuffer *b;
  while ((b = g_queue_pop_head(&context->audio_queue))){
    gst_buffer_unref(b);
  }
  if (context->audio_binner)
    recur_audio_binner_delete(context->audio_binner);
  free(context->frame_queue);
  free(context->planes);
  rnn_delete_net(context->net);
}

void
recur_context_set_video_properties(RecurContext *context, GstVideoInfo *info)
{
  context->video_info = *info;
  int expected_samples = RECUR_AUDIO_RATE * info->fps_d / info->fps_n;
  int min_window_size = ROUND_UP_4(expected_samples * 3 / 2);
  int window_size = gst_fft_next_fast_length (min_window_size);
  /* At 25 fps window_size ought to be 960; 30fps -> 800 */
  context->audio_binner = recur_audio_binner_new(window_size,
      RECUR_WINDOW_HANN,
      RECUR_N_FFT_BINS,
      RECUR_MFCC_MIN_FREQ,
      RECUR_MFCC_MAX_FREQ,
      RECUR_AUDIO_RATE,
      1.0f / (1 << 12),
      2
  );
}

/*
void
recur_context_set_audio_properties(RecurContext *context, int rate,
   int channels, GstAudioFormat format){


}
*/
