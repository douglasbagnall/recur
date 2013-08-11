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

/* Clean up */
static void
gst_parrot_finalize (GObject * obj){
  GST_DEBUG("in gst_parrot_finalize!\n");
  GstParrot *self = GST_PARROT(obj);
  recur_audio_binner_delete(self->mfcc_factory);
  rnn_delete_net(self->dream_net);
  rnn_delete_net(self->net);
  free(self->incoming_queue);
  free(self->outgoing_queue);

  mdct_clear(&self->mdct_lut);
  free(self->pcm1);
  free(self->pcm2);
  free(self->mdct1);
  free(self->mdct2);
  free(self->features1);
  free(self->features2);
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
  /*16kHz, single channel, 16 bit signed little endian PCM*/
  GstCaps *caps = gst_caps_new_simple ("audio/x-raw",
     "format", G_TYPE_STRING, PARROT_FORMAT,
     "rate", G_TYPE_INT, PARROT_RATE,
     "channels", G_TYPE_INT, PARROT_CHANNELS,
     NULL);

  gst_audio_filter_class_add_pad_templates (af_class, caps);

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
        NET_LOG_FILE, PARROT_BPTT_DEPTH, LEARN_RATE, MOMENTUM, MOMENTUM_WEIGHT);
  }
  else {
    self->net->bptt->learn_rate = LEARN_RATE;
    rnn_set_log_file(self->net, NET_LOG_FILE);
  }
  self->dream_net = rnn_clone(self->net, 0, RECUR_RNG_SUBSEED, NULL);

  self->incoming_queue = malloc_aligned_or_die(PARROT_INCOMING_QUEUE_SIZE * sizeof(s16));
  self->outgoing_queue = malloc_aligned_or_die(PARROT_OUTGOING_QUEUE_SIZE * sizeof(s16));

  mdct_init(&self->mdct_lut, PARROT_MDCT_WINDOW_SIZE);
  self->window = _vorbis_window_get(PARROT_MDCT_WINDOW_BITS - 6);
  self->pcm1 = zalloc_aligned_or_die(PARROT_MDCT_WINDOW_SIZE * sizeof(float));
  self->pcm2 = zalloc_aligned_or_die(PARROT_MDCT_WINDOW_SIZE * sizeof(float));
  self->mdct1 = zalloc_aligned_or_die(PARROT_HALF_WINDOW * sizeof(float));
  self->mdct2 = zalloc_aligned_or_die(PARROT_HALF_WINDOW * sizeof(float));
  self->features1 = zalloc_aligned_or_die(PARROT_N_FEATURES * sizeof(float));
  self->features2 = zalloc_aligned_or_die(PARROT_N_FEATURES * sizeof(float));

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
  GST_DEBUG_OBJECT (self,
      "info: %" GST_PTR_FORMAT, info);
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
  //int i;
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
    DEBUG("in possibly_save_state with generation %d", net->generation);
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
learn_and_generate(RecurNN *net, float *in, float *target){
  bptt_advance(net);
  float *answer = tanh_opinion(net, in);
  for (int i = 0; i < PARROT_HALF_WINDOW; i++){
    net->bptt->o_error[i] = target[i] - answer[i];
  }
  bptt_calculate(net);
  possibly_save_net(net);
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
  int i;
  int len_i = self->incoming_end - self->incoming_start + PARROT_INCOMING_QUEUE_SIZE;
  int len_o = self->outgoing_end - self->outgoing_start + PARROT_OUTGOING_QUEUE_SIZE;
  len_i %= PARROT_INCOMING_QUEUE_SIZE;
  len_o %= PARROT_OUTGOING_QUEUE_SIZE;
  const float *window = self->window;

  while (len_i >= PARROT_HALF_WINDOW && len_o < PARROT_OUTGOING_QUEUE_SIZE){
    s16 *buffer_i = self->incoming_queue + self->incoming_start;
    s16 *buffer_o = self->outgoing_queue + self->outgoing_end;

    /*load first half of pcm1, second part of pcm2.*/
    /*second part of pcm1 retains previous data */
    for(i = 0; i < PARROT_HALF_WINDOW; i++){
      float s = buffer_i[i] / 32768.0f;
      self->pcm1[i] = s * window[i];
      self->pcm2[PARROT_HALF_WINDOW + i] = s * window[PARROT_HALF_WINDOW - 1 - i];
    }
    /*after mdct_forward, pcm2 is free for reuse */
    mdct_forward(&self->mdct_lut, self->pcm2, self->mdct2);
    /*train. */
    float *answer;
#if PARROT_TRAIN
    pcm_to_features(self->mfcc_factory, self->features1, self->pcm1);
    answer = learn_and_generate(self->net, self->features1, self->mdct2);
#endif
#if PARROT_PASSTHROUGH
    answer = self->mdct1;
#elif PARROT_DRIFT
    answer = tanh_opinion(self->dream_net, self->features2);
#else
#endif

    mdct_backward(&self->mdct_lut, answer, self->pcm2);
    pcm_to_features(self->mfcc_factory, self->features1, self->pcm2);

    /*second half of pcm1 is still the previous frame */
    for(i = 0; i < PARROT_HALF_WINDOW; i++){
      float s = (self->pcm1[PARROT_HALF_WINDOW + i] * window[PARROT_HALF_WINDOW - 1 - i] +
          self->pcm2[i] * window[i]);
      buffer_o[i] = s * 32767.99f;
    }

    float *tmp;
    tmp = self->pcm1;
    self->pcm1 = self->pcm2;
    self->pcm2 = tmp;
    tmp = self->mdct1;
    self->mdct1 = self->mdct2;
    self->mdct2 = tmp;
    tmp = self->features1;
    self->features1 = self->features2;
    self->features2 = tmp;
    self->incoming_start += PARROT_HALF_WINDOW;
    self->incoming_start %= PARROT_INCOMING_QUEUE_SIZE;
    self->outgoing_end += PARROT_HALF_WINDOW;
    self->outgoing_end %= PARROT_INCOMING_QUEUE_SIZE;
    len_i -= PARROT_HALF_WINDOW;
    len_o += PARROT_HALF_WINDOW;
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
