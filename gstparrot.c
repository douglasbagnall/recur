/* GStreamer
 * Copyright (C) <2013> Douglas Bagnall <douglas@halo.gen.nz>
 *
 */

#include "gstparrot.h"
#include <string.h>
#include <math.h>

GST_DEBUG_CATEGORY_STATIC (parrot_debug);
#define GST_CAT_DEFAULT parrot_debug

/* GstParrot signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
};

/* static_functions */
/* plugin_init    - registers plugin (once)
   XXX_base_init  - for the gobject class (once, obsolete)
   XXX_class_init - for global state (once)
   XXX_init       - for each plugin instance
*/
static void gst_parrot_class_init(GstParrotClass *g_class);
static void gst_parrot_init(GstParrot *self);
static void gst_parrot_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_parrot_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static GstFlowReturn gst_parrot_transform(GstBaseTransform *base, GstBuffer *inbuf, GstBuffer *outbuf);
static gboolean gst_parrot_setup(GstAudioFilter * filter, const GstAudioInfo * info);

#define gst_parrot_parent_class parent_class
G_DEFINE_TYPE (GstParrot, gst_parrot, GST_TYPE_AUDIO_FILTER);

#define PARROT_CAPS_STRING "audio/x-raw, format = (string) " QUOTE(PARROT_FORMAT) \
      ", rate = (int) " QUOTE(PARROT_RATE) \
      ", channels = (int) [ " QUOTE(PARROT_MIN_CHANNELS) " , " QUOTE(PARROT_MAX_CHANNELS) " ] " \
      ", layout = (string) interleaved"


/* Clean up */
static void
gst_parrot_finalize (GObject * obj){
  GST_DEBUG("in gst_parrot_finalize!\n");
  GstParrot *self = GST_PARROT(obj);
  recur_audio_binner_delete(self->mfcc_factory);
  if (self->channels){
    for (int i = 0; i < self->n_channels; i++){
      ParrotChannel *c = &self->channels[i];
      rnn_delete_net(c->dream_net);
      rnn_delete_net(c->train_net);
      free(c->pcm1);
      free(c->pcm2);
      free(c->mdct1);
      free(c->mdct2);
      free(c->features1);
      free(c->features2);
    }
    free(self->channels);
    self->channels = NULL;
  }
  free(self->incoming_queue);
  free(self->outgoing_queue);

  mdct_clear(&self->mdct_lut);
}

static void
gst_parrot_class_init (GstParrotClass * klass)
{
  GST_DEBUG_CATEGORY_INIT (parrot_debug, "parrot", RECUR_LOG_COLOUR,
      "parrot");
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS (klass);
  GstAudioFilterClass *af_class = GST_AUDIO_FILTER_CLASS (klass);
  gobject_class->set_property = gst_parrot_set_property;
  gobject_class->get_property = gst_parrot_get_property;
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_parrot_finalize);
  /*16kHz interleaved 16 bit signed little endian PCM*/
  GstCaps *caps = gst_caps_from_string (PARROT_CAPS_STRING);
  GST_DEBUG (PARROT_CAPS_STRING);
  gst_audio_filter_class_add_pad_templates (af_class, caps);
  //free(caps);
  gst_element_class_set_static_metadata (element_class,
      "Parror audio element",
      "Filter/Audio",
      "Mangles audio",
      "Douglas Bagnall <douglas@halo.gen.nz>");

  trans_class->transform = GST_DEBUG_FUNCPTR (gst_parrot_transform);
  af_class->setup = GST_DEBUG_FUNCPTR (gst_parrot_setup);
  GST_INFO("gst audio class init\n");
}

static void
gst_parrot_init (GstParrot * self)
{

  self->net = TRY_RELOAD ? rnn_load_net(NET_FILENAME) : NULL;
  if (self->net == NULL){
    self->net = rnn_new(PARROT_N_FEATURES, PARROT_N_HIDDEN,
        PARROT_HALF_WINDOW, PARROT_RNN_FLAGS, PARROT_RNG_SEED,
        NET_LOG_FILE, PARROT_BPTT_DEPTH, LEARN_RATE, MOMENTUM, MOMENTUM_WEIGHT,
        PARROT_BATCH_SIZE);
  }
  else {
    self->net->bptt->learn_rate = LEARN_RATE;
    rnn_set_log_file(self->net, NET_LOG_FILE);
  }
  self->channels = NULL;
  self->n_channels = 0;
  self->incoming_queue = malloc_aligned_or_die(PARROT_INCOMING_QUEUE_SIZE * sizeof(s16));
  self->outgoing_queue = malloc_aligned_or_die(PARROT_OUTGOING_QUEUE_SIZE * sizeof(s16));
  self->window = _vorbis_window_get(PARROT_MDCT_WINDOW_BITS - 6);
  mdct_init(&self->mdct_lut, PARROT_MDCT_WINDOW_SIZE);

  self->mfcc_factory = recur_audio_binner_new(PARROT_MDCT_WINDOW_SIZE,
      RECUR_WINDOW_NONE,
      PARROT_N_FFT_BINS,
      PARROT_MFCC_MIN_FREQ,
      PARROT_MFCC_MAX_FREQ,
      PARROT_RATE,
      1
  );
  GST_INFO("gst parrot init\n");
}

static gboolean
gst_parrot_setup(GstAudioFilter *base, const GstAudioInfo *info){
  GST_INFO("gst_parrot_setup\n");
  GstParrot *self = GST_PARROT(base);
  //self->info = info;
  self->n_channels = info->channels;
  self->channels = malloc_aligned_or_die(self->n_channels * sizeof(ParrotChannel));
  u32 dream_flags = self->net->flags & ~(RNN_NET_FLAG_OWN_WEIGHTS | RNN_NET_FLAG_OWN_BPTT);
  u32 train_flags = self->net->flags & ~RNN_NET_FLAG_OWN_WEIGHTS;
  for (int i = 0; i < self->n_channels; i++){
    ParrotChannel *c = &self->channels[i];
    c->dream_net = rnn_clone(self->net, dream_flags, RECUR_RNG_SUBSEED, NULL);
    c->train_net = rnn_clone(self->net, train_flags, RECUR_RNG_SUBSEED, NULL);
    c->pcm1 = zalloc_aligned_or_die(PARROT_MDCT_WINDOW_SIZE * sizeof(float));
    c->pcm2 = zalloc_aligned_or_die(PARROT_MDCT_WINDOW_SIZE * sizeof(float));
    c->mdct1 = zalloc_aligned_or_die(PARROT_HALF_WINDOW * sizeof(float));
    c->mdct2 = zalloc_aligned_or_die(PARROT_HALF_WINDOW * sizeof(float));
    c->features1 = zalloc_aligned_or_die(PARROT_N_FEATURES * sizeof(float));
    c->features2 = zalloc_aligned_or_die(PARROT_N_FEATURES * sizeof(float));
  }
  GST_DEBUG_OBJECT (self,
      "info: %" GST_PTR_FORMAT, info);
  DEBUG("found %d channels", self->n_channels);
  return TRUE;
}

static inline void
set_string_prop(const GValue *value, const char **target){
  const char *s = g_value_dup_string(value);
  size_t len = strlen(s);
  if(len){
    *target = s;
  }
}

static void
gst_parrot_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  //GstParrot *self = GST_PARROT (object);
  GST_DEBUG("gst_parrot_set_property\n");
  if (value){
    switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
    }
  }
}

static void
gst_parrot_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  //GstParrot *self = GST_PARROT (object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static inline void
queue_audio_segment(GstParrot *self, GstBuffer *inbuf)
{
  GstMapInfo map;
  //XXX deinterleave.
  gst_buffer_map(inbuf, &map, 0);
  //s16 *audio = (s16 *)map.data;
  size_t len = map.size / sizeof(s16);
  int end = self->incoming_end;
  if (end + len < PARROT_INCOMING_QUEUE_SIZE){
    memcpy(self->incoming_queue + end, map.data, map.size);
    self->incoming_end += len;
  }
  else {
    int snip = PARROT_INCOMING_QUEUE_SIZE - end;
    int snip8 = snip * sizeof(s16);
    memcpy(self->incoming_queue + end, map.data, snip8);
    memcpy(self->incoming_queue, map.data + snip8,
        map.size - snip8);
    self->incoming_end = len - snip;
  }

  int lag = self->incoming_end - self->incoming_start;
  if (lag < 0){
    lag += PARROT_INCOMING_QUEUE_SIZE;
  }
  if (lag + len > PARROT_INCOMING_QUEUE_SIZE){
    GST_DEBUG("incoming lag %d seems to exceed queue size %d",
        lag, PARROT_INCOMING_QUEUE_SIZE);
  }
  GST_LOG("queueing audio starting %llu, ending %llu",
      GST_BUFFER_PTS(inbuf), GST_BUFFER_PTS(inbuf) + GST_BUFFER_DURATION(inbuf));
  gst_buffer_unmap(inbuf, &map);
}

static inline void
fill_audio_segment(GstParrot *self, GstBuffer *outbuf)
{
  GstMapInfo map;
  gst_buffer_map(outbuf, &map, GST_MAP_WRITE);
  int len16 = map.size / sizeof(s16);
  int start = self->outgoing_start;
  int end = self->outgoing_end;
  int avail = (end >= start) ? end - start : end - start + PARROT_OUTGOING_QUEUE_SIZE;
  if (avail < len16){
    GST_INFO("insufficient audio! want %d, have %d, sending zeros", len16, avail);
    memset(map.data, 0, map.size);
    return;
  }

  if (start + len16 < PARROT_INCOMING_QUEUE_SIZE){
    memcpy(map.data, self->outgoing_queue + start, map.size);
    self->outgoing_start += len16;
  }
  else {
    int snip = PARROT_INCOMING_QUEUE_SIZE - start;
    int snip8 = snip * sizeof(s16);
    memcpy(map.data, self->outgoing_queue + start, snip8);
    memcpy(map.data + snip8, self->outgoing_queue,
        map.size - snip8);
    self->outgoing_start += len16 - PARROT_INCOMING_QUEUE_SIZE;
  }
  gst_buffer_unmap(outbuf, &map);
}


static inline void
possibly_save_net(RecurNN *net)
{
  if (PERIODIC_SAVE_NET && (net->generation & 511) == 0){
    rnn_save_net(net, NET_FILENAME);
  }
  if (REGULAR_PGM_DUMP)
    rnn_multi_pgm_dump(net, "ihw hhw");
  else if (PERIODIC_PGM_DUMP && net->generation % PERIODIC_PGM_DUMP == 0)
    rnn_multi_pgm_dump(net, "hhw ihw");
}



static inline float *
tanh_opinion(RecurNN *net, float *in){
  float *answer = rnn_opinion(net, in);
  for (int i = 0; i < PARROT_HALF_WINDOW; i++){
    answer[i] = fast_tanhf(answer[i]);
  }
  return answer;
}

static inline float*
generate_and_update_error(RecurNN *net, float *in, float *target, float *error){
  //bptt_advance(net);
  float *answer = tanh_opinion(net, in);
  for (int i = 0; i < PARROT_HALF_WINDOW; i++){
    error[i] += target[i] - answer[i];
  }
  return answer;
}


static inline void
pcm_to_features(RecurAudioBinner *mf, float *features, float *pcm){
#if PARROT_USE_MFCCS
  recur_extract_mfccs(mf, pcm);
#else
  recur_extract_log_freq_bins(mf, pcm);
#endif
  for (int i = 0; i < PARROT_N_FEATURES; i++){
    features[i] = mf->dct_bins[i];
  }
}

static inline void
maybe_learn_and_generate_audio(GstParrot *self){
  int i, j, k;
  int len_i = self->incoming_end - self->incoming_start + PARROT_INCOMING_QUEUE_SIZE;
  int len_o = self->outgoing_end - self->outgoing_start + PARROT_OUTGOING_QUEUE_SIZE;
  len_i %= PARROT_INCOMING_QUEUE_SIZE;
  len_o %= PARROT_OUTGOING_QUEUE_SIZE;
  const float *window = self->window;
  int chunk_size = PARROT_HALF_WINDOW * self->n_channels;

  while (len_i >= chunk_size && len_o < PARROT_OUTGOING_QUEUE_SIZE){
    s16 *buffer_i = self->incoming_queue + self->incoming_start;
    s16 *buffer_o = self->outgoing_queue + self->outgoing_end;

#if PARROT_TRAIN
    //bptt_advance(self->net);
    float * error = self->net->bptt->o_error;
    memset(error, 0, self->net->o_size * sizeof(float));
#endif

    for (j = 0; j < self->n_channels; j++){
      /*load first half of pcm1, second part of pcm2.*/
      /*second part of pcm1 retains previous data */
      ParrotChannel *c = & self->channels[j];
      for(i = 0, k = j; i < PARROT_HALF_WINDOW; i++, k += self->n_channels){
        float s = buffer_i[k] / 32768.0f;
        c->pcm1[i] = s * window[i];
        c->pcm2[PARROT_HALF_WINDOW + i] = s * window[PARROT_HALF_WINDOW - 1 - i];
      }

      /*after mdct_forward, pcm2 is free for reuse */
      mdct_forward(&self->mdct_lut, c->pcm2, c->mdct2);
      /*train. */
      float *answer;
#if PARROT_TRAIN
      bptt_advance(c->train_net);
      pcm_to_features(self->mfcc_factory, c->features1, c->pcm1);
      answer = generate_and_update_error(c->train_net, c->features1, c->mdct2, error);
      bptt_calc_deltas(c->train_net);
      //for (int i = 0; i < PARROT_HALF_WINDOW; i++){
      //  error[i] += c->features1[i] - c->mdct2[i];
      //}
#endif
#if PARROT_PASSTHROUGH
      answer = c->mdct1;
#elif PARROT_DRIFT
      answer = tanh_opinion(c->dream_net, c->features2);
#else
#endif
      mdct_backward(&self->mdct_lut, answer, c->pcm2);
      pcm_to_features(self->mfcc_factory, c->features1, c->pcm2);

      /*second half of pcm1 is still the previous frame */
      for(i = 0, k = j; i < PARROT_HALF_WINDOW; i++, k += self->n_channels){
        float s = (c->pcm1[PARROT_HALF_WINDOW + i] * window[PARROT_HALF_WINDOW - 1 - i] +
            c->pcm2[i] * window[i]);
        buffer_o[k] = s * 32767.99f;
      }

      float *tmp;
      tmp = c->pcm1;
      c->pcm1 = c->pcm2;
      c->pcm2 = tmp;
      tmp = c->mdct1;
      c->mdct1 = c->mdct2;
      c->mdct2 = tmp;
      tmp = c->features1;
      c->features1 = c->features2;
      c->features2 = tmp;
    }

#if PARROT_TRAIN
    //bptt_calculate(self->net);
    RecurNN *nets[self->n_channels];
    for (int j = 0; j < self->n_channels; j++){
      nets[j] = self->channels[j].train_net;
    }
    bptt_consolidate_many_nets(nets, self->n_channels);
    possibly_save_net(self->net);
#endif
    self->incoming_start += chunk_size;
    self->incoming_start %= PARROT_INCOMING_QUEUE_SIZE;
    self->outgoing_end += chunk_size;
    self->outgoing_end %= PARROT_INCOMING_QUEUE_SIZE;
    len_i -= chunk_size;
    len_o += chunk_size;
  }
}

static GstFlowReturn
gst_parrot_transform (GstBaseTransform * base, GstBuffer *inbuf, GstBuffer *outbuf)
{
  GstParrot *self = GST_PARROT(base);
  GstFlowReturn ret = GST_FLOW_OK;

  queue_audio_segment(self, inbuf);
  maybe_learn_and_generate_audio(self);
  fill_audio_segment(self, outbuf);

  GST_LOG("parrot_transform returning OK");
  //exit:
  return ret;
}


static gboolean
plugin_init (GstPlugin * plugin)
{
  GST_INFO("parrot plugin init\n");
  gboolean parrot = gst_element_register (plugin, "parrot", GST_RANK_NONE,\
      GST_TYPE_PARROT);
  return parrot;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    parrot,
    "Parrot audio streams",
    plugin_init, VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN);
