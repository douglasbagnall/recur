/* GStreamer
 * Copyright (C) <2013> Douglas Bagnall <douglas@halo.gen.nz>
 *
 */

#include "gstclassify.h"
#include <string.h>
#include <math.h>

GST_DEBUG_CATEGORY_STATIC (classify_debug);
#define GST_CAT_DEFAULT classify_debug

/* GstClassify signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_TARGET,
};

#define DEFAULT_PROP_TARGET ""


/* static_functions */
/* plugin_init    - registers plugin (once)
   XXX_base_init  - for the gobject class (once, obsolete)
   XXX_class_init - for global state (once)
   XXX_init       - for each plugin instance
*/
static void gst_classify_class_init(GstClassifyClass *g_class);
static void gst_classify_init(GstClassify *self);
static void gst_classify_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_classify_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static GstFlowReturn gst_classify_transform_ip(GstBaseTransform *base, GstBuffer *buf);
static gboolean gst_classify_setup(GstAudioFilter * filter, const GstAudioInfo * info);

#define gst_classify_parent_class parent_class
G_DEFINE_TYPE (GstClassify, gst_classify, GST_TYPE_AUDIO_FILTER);

#define CLASSIFY_CAPS_STRING "audio/x-raw, format = (string) " QUOTE(CLASSIFY_FORMAT) \
      ", rate = (int) " QUOTE(CLASSIFY_RATE) \
      ", channels = (int) [ " QUOTE(CLASSIFY_MIN_CHANNELS) " , " QUOTE(CLASSIFY_MAX_CHANNELS) " ] " \
      ", layout = (string) interleaved"


/* Clean up */
static void
gst_classify_finalize (GObject * obj){
  GST_DEBUG("in gst_classify_finalize!\n");
  GstClassify *self = GST_CLASSIFY(obj);
  recur_audio_binner_delete(self->mfcc_factory);
  if (self->channels){
    for (int i = 0; i < self->n_channels; i++){
      ClassifyChannel *c = &self->channels[i];
      rnn_delete_net(c->train_net);
      free(c->pcm_next);
      free(c->pcm_now);
      free(c->features);
    }
    free(self->channels);
    self->channels = NULL;
  }
  free(self->incoming_queue);
}

static void
gst_classify_class_init (GstClassifyClass * klass)
{
  GST_DEBUG_CATEGORY_INIT (classify_debug, "classify", RECUR_LOG_COLOUR,
      "classify");
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
  GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS (klass);
  GstAudioFilterClass *af_class = GST_AUDIO_FILTER_CLASS (klass);
  gobject_class->set_property = gst_classify_set_property;
  gobject_class->get_property = gst_classify_get_property;
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_classify_finalize);
  /*8kHz interleaved 16 bit signed little endian PCM*/
  GstCaps *caps = gst_caps_from_string (CLASSIFY_CAPS_STRING);
  GST_DEBUG (CLASSIFY_CAPS_STRING);
  gst_audio_filter_class_add_pad_templates (af_class, caps);
  //free(caps);
  gst_element_class_set_static_metadata (element_class,
      "Parror audio element",
      "Filter/Audio",
      "Mangles audio",
      "Douglas Bagnall <douglas@halo.gen.nz>");

  g_object_class_install_property (gobject_class, PROP_TARGET,
      g_param_spec_string("target", "target",
          "Target outputs for all channels (dot separated)",
          DEFAULT_PROP_TARGET,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  trans_class->transform_ip = GST_DEBUG_FUNCPTR (gst_classify_transform_ip);
  af_class->setup = GST_DEBUG_FUNCPTR (gst_classify_setup);
  GST_INFO("gst audio class init\n");
}

static void
gst_classify_init (GstClassify * self)
{

  self->net = TRY_RELOAD ? rnn_load_net(NET_FILENAME) : NULL;
  if (self->net == NULL){
    self->net = rnn_new(CLASSIFY_N_FEATURES, CLASSIFY_N_HIDDEN,
        CLASSIFY_HALF_WINDOW, CLASSIFY_RNN_FLAGS, CLASSIFY_RNG_SEED,
        NET_LOG_FILE, CLASSIFY_BPTT_DEPTH, LEARN_RATE, MOMENTUM, MOMENTUM_WEIGHT,
        CLASSIFY_BATCH_SIZE);
  }
  else {
    self->net->bptt->learn_rate = LEARN_RATE;
    rnn_set_log_file(self->net, NET_LOG_FILE);
  }
  self->channels = NULL;
  self->n_channels = 0;
  self->incoming_queue = malloc_aligned_or_die(CLASSIFY_INCOMING_QUEUE_SIZE * sizeof(s16));

  self->mfcc_factory = recur_audio_binner_new(CLASSIFY_WINDOW_SIZE,
      RECUR_WINDOW_NONE,
      CLASSIFY_N_FFT_BINS,
      CLASSIFY_MFCC_MIN_FREQ,
      CLASSIFY_MFCC_MAX_FREQ,
      CLASSIFY_RATE,
      1
  );
  GST_INFO("gst classify init\n");
}

static gboolean
gst_classify_setup(GstAudioFilter *base, const GstAudioInfo *info){
  GST_INFO("gst_classify_setup\n");
  GstClassify *self = GST_CLASSIFY(base);
  //self->info = info;
  self->n_channels = info->channels;
  self->channels = malloc_aligned_or_die(self->n_channels * sizeof(ClassifyChannel));
  u32 train_flags = self->net->flags & ~RNN_NET_FLAG_OWN_WEIGHTS;
  for (int i = 0; i < self->n_channels; i++){
    ClassifyChannel *c = &self->channels[i];
    c->train_net = rnn_clone(self->net, train_flags, RECUR_RNG_SUBSEED, NULL);
    c->pcm_next = zalloc_aligned_or_die(CLASSIFY_WINDOW_SIZE * sizeof(float));
    c->pcm_now = zalloc_aligned_or_die(CLASSIFY_WINDOW_SIZE * sizeof(float));
    c->features = zalloc_aligned_or_die(CLASSIFY_N_FEATURES * sizeof(float));
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

static inline void
parse_target_string(GstClassify *self, const char *s){
  int i;
  if (s == NULL){
    return;
  }
  if (*s == 0){
    for (i = 0; i < self->n_channels; i++){
      self->channels[i].current_target = -1;
    }
    self->training = 0;
    return;
  }
  const char *orig = s;
  char *e;
  long x;
  for (i = 0; i < self->n_channels; i++){
    x = strtol(s, &e, 10);
    if (s == e){ /*no digits at all */
      goto parse_error;
    }
    self->channels[i].current_target = x;
    s = e + 1;
    if (*e == '\0'  && i != self->n_channels - 1){ /*ran out of numbers */
      goto parse_error;
    }
  }
  return;
 parse_error:
  GST_DEBUG("Can't parse '%s' into %d channels: stopping after %d",
      orig, self->n_channels, i);
  return;
}

static void
gst_classify_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstClassify *self = GST_CLASSIFY (object);
  GST_DEBUG("gst_classify_set_property\n");
  if (value){
    switch (prop_id) {
    case PROP_TARGET:
      parse_target_string(self, g_value_get_string(value));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
    }
  }
}

static void
gst_classify_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstClassify *self = GST_CLASSIFY (object);
  char s[100];
  char *t = s;
  int remaining = sizeof(s) - 2;
  int wrote;
  int i;
  switch (prop_id) {
    case PROP_TARGET:
      for (i = 0; i < self->n_channels; i++){
        wrote = snprintf(t, remaining, "%d.", self->channels[i].current_target);
        remaining -= wrote;
        t += wrote;
        if (remaining < 11)
          break;
      }
      *t = 0;
      g_value_set_string(value, s);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* queue_audio_segment collects up the audio data, but leaves the
   deinterlacing and interpretation till later. This is a decoupling step
   because the incoming packet size is unrelated to the evaluation window
   size.
 */
static inline void
queue_audio_segment(GstClassify *self, GstBuffer *inbuf)
{
  GstMapInfo map;
  gst_buffer_map(inbuf, &map, 0);
  size_t len = map.size / sizeof(s16);
  int end = self->incoming_end;
  if (end + len < CLASSIFY_INCOMING_QUEUE_SIZE){
    memcpy(self->incoming_queue + end, map.data, map.size);
    self->incoming_end += len;
  }
  else {
    int snip = CLASSIFY_INCOMING_QUEUE_SIZE - end;
    int snip8 = snip * sizeof(s16);
    memcpy(self->incoming_queue + end, map.data, snip8);
    memcpy(self->incoming_queue, map.data + snip8,
        map.size - snip8);
    self->incoming_end = len - snip;
  }

  int lag = self->incoming_end - self->incoming_start;
  if (lag < 0){
    lag += CLASSIFY_INCOMING_QUEUE_SIZE;
  }
  if (lag + len > CLASSIFY_INCOMING_QUEUE_SIZE){
    GST_DEBUG("incoming lag %d seems to exceed queue size %d",
        lag, CLASSIFY_INCOMING_QUEUE_SIZE);
  }
  GST_LOG("queueing audio starting %llu, ending %llu",
      GST_BUFFER_PTS(inbuf), GST_BUFFER_PTS(inbuf) + GST_BUFFER_DURATION(inbuf));
  gst_buffer_unmap(inbuf, &map);
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

/*
static GstMessage *
prepare_message (GstClassify *self)
{
  char *name = "message"
  GstStructure *s;
  s = gst_structure_new (name,
      "type", G_TYPE_INT, 1,
      "method", G_TYPE_INT, 1,
      "start", G_TYPE_BOOLEAN, TRUE,
      "number", G_TYPE_INT, event->payload->event,
      "volume", G_TYPE_INT, event->payload->volume, NULL);

  return gst_message_new_element (GST_OBJECT (dtmfsrc), s);
}

*/



static inline void
pcm_to_features(RecurAudioBinner *mf, float *features, float *pcm){
#if CLASSIFY_USE_MFCCS
  recur_extract_mfccs(mf, pcm);
#else
  recur_extract_log_freq_bins(mf, pcm);
#endif
  for (int i = 0; i < CLASSIFY_N_FEATURES; i++){
    features[i] = mf->dct_bins[i];
  }
}

static inline void
maybe_learn(GstClassify *self){
  int i, j, k;
  int len = self->incoming_end - self->incoming_start;
  if (len < 0){ //XXX or less than or equal?
    len += CLASSIFY_INCOMING_QUEUE_SIZE;
  }
  int chunk_size = CLASSIFY_HALF_WINDOW * self->n_channels;

  while (len >= chunk_size){
    s16 *buffer_i = self->incoming_queue + self->incoming_start;

    if (self->training){
      float *error = self->net->bptt->o_error;
      memset(error, 0, self->net->o_size * sizeof(float));
    }

    for (j = 0; j < self->n_channels; j++){
      /*load first half of pcm_next, second part of pcm_now.*/
      /*second part of pcm_next retains previous data */
      ClassifyChannel *c = & self->channels[j];
      RecurNN *net = c->train_net;
      for(i = 0, k = j; i < CLASSIFY_HALF_WINDOW; i++, k += self->n_channels){
        float s = buffer_i[k] / 32768.0f;
        c->pcm_next[i] = s;
        c->pcm_now[CLASSIFY_HALF_WINDOW + i] = s;
      }

      /*get the features -- after which pcm_now is finished with. */
      pcm_to_features(self->mfcc_factory, c->features, c->pcm_now);

      float *answer;

      if (self->training){
        bptt_advance(net);
      }
      answer = rnn_opinion(net, c->features);
      int correct;
      float err = softmax_best_guess(net->bptt->o_error, answer,
          net->output_size, c->current_target, &correct);
      if (self->training){
        bptt_calc_deltas(net);
      }
      float *tmp;
      tmp = c->pcm_next;
      c->pcm_next = c->pcm_now;
      c->pcm_now = tmp;
    }

    if (self->training){
      //bptt_calculate(self->net);
      RecurNN *nets[self->n_channels];
      for (int j = 0; j < self->n_channels; j++){
        nets[j] = self->channels[j].train_net;
      }
      bptt_consolidate_many_nets(nets, self->n_channels);
      possibly_save_net(self->net);
    }

    self->incoming_start += chunk_size;
    self->incoming_start %= CLASSIFY_INCOMING_QUEUE_SIZE;
    len -= chunk_size;
  }
}

static GstFlowReturn
gst_classify_transform_ip (GstBaseTransform * base, GstBuffer *buf)
{
  GstClassify *self = GST_CLASSIFY(base);
  GstFlowReturn ret = GST_FLOW_OK;
  queue_audio_segment(self, buf);
  maybe_learn(self);
  GST_LOG("classify_transform returning OK");
  return ret;
}


static gboolean
plugin_init (GstPlugin * plugin)
{
  GST_INFO("classify plugin init\n");
  gboolean classify = gst_element_register (plugin, "classify", GST_RANK_NONE,\
      GST_TYPE_CLASSIFY);
  return classify;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    classify,
    "Classify audio streams",
    plugin_init, VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN);
