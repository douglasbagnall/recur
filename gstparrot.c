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
  PROP_FORGET,
  PROP_LEARN_RATE,
  PROP_HIDDEN_SIZE,
  PROP_SAVE_NET,
  PROP_PGM_DUMP,
  PROP_LOG_FILE,
};

#define DEFAULT_PROP_PGM_DUMP ""
#define DEFAULT_PROP_LOG_FILE ""
#define DEFAULT_PROP_SAVE_NET NULL
#define DEFAULT_PROP_FORGET 0
#define DEFAULT_HIDDEN_SIZE 199
#define DEFAULT_LEARN_RATE 0.0001
#define MIN_HIDDEN_SIZE 1
#define MAX_HIDDEN_SIZE 1000000
#define LEARN_RATE_MIN 0.0
#define LEARN_RATE_MAX 1.0

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
static void maybe_start_logging(GstParrot *self);


#define gst_parrot_parent_class parent_class
G_DEFINE_TYPE (GstParrot, gst_parrot, GST_TYPE_AUDIO_FILTER);

#define PARROT_CAPS_STRING "audio/x-raw, format = (string) " QUOTE(PARROT_FORMAT) \
      ", rate = (int) " QUOTE(PARROT_RATE) \
      ", channels = (int) [ " QUOTE(PARROT_MIN_CHANNELS) " , " QUOTE(PARROT_MAX_CHANNELS) " ] " \
                                                                                          ", layout = (string) interleaved"/*", channel-mask = (bitmask)0x0"*/

static inline void
init_channel(ParrotChannel *c, RecurNN *net, int id, float learn_rate)
{
  u32 train_flags = net->flags & ~RNN_NET_FLAG_OWN_WEIGHTS;
  u32 dream_flags = net->flags & ~(RNN_NET_FLAG_OWN_WEIGHTS | RNN_NET_FLAG_OWN_BPTT);
  c->train_net = rnn_clone(net, train_flags, RECUR_RNG_SUBSEED, NULL);
  c->dream_net = rnn_clone(net, dream_flags, RECUR_RNG_SUBSEED, NULL);
  c->train_net->bptt->learn_rate = learn_rate;
  c->pcm_now = zalloc_aligned_or_die(PARROT_WINDOW_SIZE * sizeof(float));
  c->pcm_prev = zalloc_aligned_or_die(PARROT_WINDOW_SIZE * sizeof(float));
  c->play_now = zalloc_aligned_or_die(PARROT_WINDOW_SIZE * sizeof(float));
  c->play_prev = zalloc_aligned_or_die(PARROT_WINDOW_SIZE * sizeof(float));
  c->mdct_target = zalloc_aligned_or_die(PARROT_WINDOW_SIZE * sizeof(float));
  c->features = zalloc_aligned_or_die(PARROT_N_FEATURES * sizeof(float));
  if (PGM_DUMP_FEATURES){
    c->mfcc_image = temporal_ppm_alloc(PARROT_N_FEATURES, 300, "parrot-mfcc", id,
        PGM_DUMP_COLOUR);
  }
  else {
    c->mfcc_image = NULL;
  }
}

static inline void
finalise_channel(ParrotChannel *c)
{
  rnn_delete_net(c->train_net);
  rnn_delete_net(c->dream_net);
  free(c->pcm_prev);
  free(c->pcm_now);
  free(c->mdct_target);
  free(c->features);
  if (c->mfcc_image){
    temporal_ppm_free(c->mfcc_image);
    c->mfcc_image = NULL;
  }
}

/* Clean up */
static void
gst_parrot_finalize (GObject * obj){
  GST_DEBUG("in gst_parrot_finalize!\n");
  GstParrot *self = GST_PARROT(obj);
  if (self->net){
    rnn_save_net(self->net, self->net_filename);
  }
  if (self->mfcc_factory){
    recur_audio_binner_delete(self->mfcc_factory);
  }
  if (self->channels){
    for (int i = 0; i < self->n_channels; i++){
      finalise_channel(&self->channels[i]);
      self->training_nets[i] = NULL;
    }
    free(self->channels);
    self->channels = NULL;
    free(self->training_nets);
  }
  free(self->incoming_queue);
  free(self->outgoing_queue);

  mdct_clear(&self->mdct_lut);
  if (self->net){
    rnn_delete_net(self->net);
  }
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
      "Parrot audio element",
      "Filter/Audio",
      "Parrots audio",
      "Douglas Bagnall <douglas@halo.gen.nz>");

  g_object_class_install_property (gobject_class, PROP_PGM_DUMP,
      g_param_spec_string("pgm-dump", "pgm-dump",
          "Dump weight images (space separated \"ih* hh* ho*\", *one of \"wdm\")",
          DEFAULT_PROP_PGM_DUMP,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SAVE_NET,
      g_param_spec_string("save-net", "save-net",
          "Save the net here, now.",
          DEFAULT_PROP_SAVE_NET,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LOG_FILE,
      g_param_spec_string("log-file", "log-file",
          "Log to this file (empty for none)",
          DEFAULT_PROP_LOG_FILE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_FORGET,
      g_param_spec_boolean("forget", "forget",
          "Forget the current hidden layer (all channels)",
          DEFAULT_PROP_FORGET,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LEARN_RATE,
      g_param_spec_float("learn-rate", "learn-rate",
          "Learning rate for the RNN",
          LEARN_RATE_MIN, LEARN_RATE_MAX,
          DEFAULT_LEARN_RATE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_HIDDEN_SIZE,
      g_param_spec_int("hidden-size", "hidden-size",
          "Size of the RNN hidden layer",
          MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE,
          DEFAULT_HIDDEN_SIZE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));


  trans_class->transform = GST_DEBUG_FUNCPTR (gst_parrot_transform);
  af_class->setup = GST_DEBUG_FUNCPTR (gst_parrot_setup);
  GST_INFO("gst audio class init\n");
}

static void
gst_parrot_init (GstParrot * self)
{
  self->channels = NULL;
  self->n_channels = 0;
  self->mfcc_factory = NULL;
  self->incoming_queue = NULL;
  self->net_filename = NULL;
  self->pending_logfile = NULL;
  self->learn_rate = DEFAULT_LEARN_RATE;
  self->hidden_size = DEFAULT_HIDDEN_SIZE;
  /*XXX add switches */
  self->training = 1;
  self->playing = 1;
  GST_INFO("gst parrot init\n");
}

static void
reset_net_filename(GstParrot *self){
  char s[200];
  snprintf(s, sizeof(s), "parrot-i%d-h%d-o%d-b%d-%dHz.net",
      PARROT_N_FEATURES, self->hidden_size, PARROT_WINDOW_SIZE,
      PARROT_BIAS, PARROT_RATE);

  if (self->net_filename){
    if (! streq(s, self->net_filename)){
      free(self->net_filename);
      self->net_filename = strdup(s);
    }
  }
  else {
    self->net_filename = strdup(s);
  }
}

static RecurNN *
load_or_create_net(GstParrot *self){
  reset_net_filename(self);
  RecurNN *net = TRY_RELOAD ? rnn_load_net(self->net_filename) : NULL;
  if (net){
    if (net->output_size != PARROT_WINDOW_SIZE){
      GST_WARNING("loaded net doesn't seem to match!");
      rnn_delete_net(net);
      net = NULL;
    }
  }
  if (net == NULL){
    net = rnn_new(PARROT_N_FEATURES, self->hidden_size,
        PARROT_WINDOW_SIZE, PARROT_RNN_FLAGS, PARROT_RNG_SEED,
        NULL, PARROT_BPTT_DEPTH, self->learn_rate, MOMENTUM, MOMENTUM_WEIGHT,
        PARROT_BATCH_SIZE);
  }
  else {
    rnn_set_log_file(net, NULL, 0);
  }
  return net;
}


static gboolean
gst_parrot_setup(GstAudioFilter *base, const GstAudioInfo *info){
  GST_INFO("gst_parrot_setup\n");
  GstParrot *self = GST_PARROT(base);
  self->n_channels = info->channels;
  if (self->incoming_queue == NULL){
    self->queue_size = info->channels * PARROT_QUEUE_PER_CHANNEL;
    self->incoming_queue = malloc_aligned_or_die(self->queue_size * sizeof(s16) * 2);
    self->outgoing_queue = self->incoming_queue + self->queue_size * sizeof(s16);
  }
  self->window = malloc_aligned_or_die(PARROT_WINDOW_SIZE * sizeof(float));
  recur_window_init(self->window, PARROT_WINDOW_SIZE,
      RECUR_WINDOW_VORBIS, 0.9999f / (1<<15));

  mdct_init(&self->mdct_lut, PARROT_WINDOW_SIZE);

  if (self->mfcc_factory == NULL){
    self->mfcc_factory = recur_audio_binner_new(PARROT_WINDOW_SIZE,
        RECUR_WINDOW_HANN,
        PARROT_N_FFT_BINS,
        PARROT_MFCC_MIN_FREQ,
        PARROT_MFCC_MAX_FREQ,
        PARROT_RATE,
        1.0f / 32768,
        PARROT_VALUE_SIZE
    );
  }
  if (self->net == NULL){
    self->net = load_or_create_net(self);
  }
  if (self->channels == NULL){
    self->channels = malloc_aligned_or_die(self->n_channels * sizeof(ParrotChannel));
    self->training_nets = malloc_aligned_or_die(self->n_channels * sizeof(RecurNN *));
    for (int i = 0; i < self->n_channels; i++){
      init_channel(&self->channels[i], self->net, i, self->learn_rate);
      self->training_nets[i] = self->channels[i].train_net;
    }
  }
  maybe_start_logging(self);

  GST_DEBUG_OBJECT (self,
      "info: %" GST_PTR_FORMAT, info);
  DEBUG("found %d channels", self->n_channels);
  return TRUE;
}

static void
maybe_start_logging(GstParrot *self){
  if (self->pending_logfile && self->training_nets){
    if (self->pending_logfile[0] == 0){
      rnn_set_log_file(self->training_nets[0], NULL, 0);
    }
    else {
      rnn_set_log_file(self->training_nets[0], self->pending_logfile, 1);
    }
    free(self->pending_logfile);
    self->pending_logfile = NULL;
  }
}


static void
gst_parrot_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstParrot *self = GST_PARROT (object);
  GST_DEBUG("gst_parrot_set_property\n");
  if (value){
    const char *strvalue;
    switch (prop_id) {
    case PROP_PGM_DUMP:
      strvalue = g_value_get_string(value);
      rnn_multi_pgm_dump(self->net, strvalue);
      break;

    case PROP_SAVE_NET:
      strvalue = g_value_get_string(value);
      if (strvalue && strvalue[0] != 0){
        rnn_save_net(self->net, strvalue);
      }
      else {
        rnn_save_net(self->net, self->net_filename);
      }
      break;

    case PROP_LOG_FILE:
      /*defer setting the actual log file, in case the nets aren't ready yet*/
      if (self->pending_logfile){
        free(self->pending_logfile);
      }
      self->pending_logfile = g_value_dup_string(value);
      maybe_start_logging(self);
      break;

    case PROP_FORGET:
      if (self->net){
        gboolean bptt_too = g_value_get_boolean(value);
        rnn_forget_history(self->net, bptt_too);
        for (int i = 0; i < self->n_channels; i++){
          rnn_forget_history(self->channels[i].train_net, bptt_too);
        }
      }
      break;

    case PROP_LEARN_RATE:
      self->learn_rate = g_value_get_float(value);
      if (self->net){
        float learn_rate = g_value_get_float(value);
        for (int i = 0; i < self->n_channels; i++){
          self->channels[i].train_net->bptt->learn_rate = learn_rate;
        }
      }
      break;

    case PROP_HIDDEN_SIZE:
      self->hidden_size = g_value_get_int(value);
      if (self->net){
        GST_WARNING("It is too late to set hidden size");
      }
      break;

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
  GstParrot *self = GST_PARROT (object);
  switch (prop_id) {

  case PROP_LEARN_RATE:
    g_value_set_float(value, self->learn_rate);
    break;

  case PROP_HIDDEN_SIZE:
    g_value_set_int(value, self->hidden_size);
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
queue_audio_segment(GstParrot *self, GstBuffer *inbuf)
{
  GstMapInfo map;
  gst_buffer_map(inbuf, &map, 0);
  int len = map.size / sizeof(s16);
  int end = self->incoming_end;
  if (end + len < self->queue_size){
    memcpy(self->incoming_queue + end, map.data, map.size);
    self->incoming_end += len;
  }
  else {
    int snip = self->queue_size - end;
    int snip8 = snip * sizeof(s16);
    memcpy(self->incoming_queue + end, map.data, snip8);
    memcpy(self->incoming_queue, map.data + snip8,
        map.size - snip8);
    self->incoming_end = len - snip;
  }

  int lag = self->incoming_end - self->incoming_start;
  if (lag < 0){
    lag += self->queue_size;
  }
  if (lag + len > self->queue_size){
    GST_DEBUG("incoming lag %d seems to exceed queue size %d",
        lag, self->queue_size);
  }
  GST_LOG("queueing audio starting %llu, ending %llu",
      GST_BUFFER_PTS(inbuf), GST_BUFFER_PTS(inbuf) + GST_BUFFER_DURATION(inbuf));
  gst_buffer_unmap(inbuf, &map);
}

static inline void
fill_audio_segment(GstParrot *self, GstBuffer *outbuf)
{
  /*XXX interlace */
  GstMapInfo map;
  gst_buffer_map(outbuf, &map, GST_MAP_WRITE);
  int len16 = map.size / sizeof(s16);
  int start = self->outgoing_start;
  int end = self->outgoing_end;
  int avail = (end >= start) ? end - start : end - start + self->queue_size;
  if (avail < len16){
    GST_INFO("insufficient audio! want %d, have %d, sending zeros", len16, avail);
    memset(map.data, 0, map.size);
    return;
  }

  if (start + len16 < self->queue_size){
    memcpy(map.data, self->outgoing_queue + start, map.size);
    self->outgoing_start += len16;
  }
  else {
    int snip = self->queue_size - start;
    int snip8 = snip * sizeof(s16);
    memcpy(map.data, self->outgoing_queue + start, snip8);
    memcpy(map.data + snip8, self->outgoing_queue,
        map.size - snip8);
    self->outgoing_start += len16 - self->queue_size;
  }
  gst_buffer_unmap(outbuf, &map);
}

static inline void
possibly_save_net(RecurNN *net, char *filename)
{
  GST_LOG("possibly saving to %s", filename);
  if (PERIODIC_SAVE_NET && (net->generation & 511) == 0){
    rnn_save_net(net, filename);
  }
  if (REGULAR_PGM_DUMP)
    rnn_multi_pgm_dump(net, "ihw hhw");
  else if (PERIODIC_PGM_DUMP && net->generation % PERIODIC_PGM_DUMP == 0)
    rnn_multi_pgm_dump(net, "hhw ihw");
}

static inline void
pcm_to_features(RecurAudioBinner *mf, float *features, float *pcm){
  float *answer;
#if PARROT_USE_MFCCS
  answer = recur_extract_mfccs(mf, pcm);
#else
  answer = recur_extract_log_freq_bins(mf, pcm);
#endif
  for (int i = 0; i < PARROT_N_FEATURES; i++){
    features[i] = answer[i];
  }
}


static inline float *
tanh_opinion(RecurNN *net, float *in){
  float *answer = rnn_opinion(net, in);
  for (int i = 0; i < (PARROT_WINDOW_SIZE / 2); i++){
    answer[i] = fast_tanhf(answer[i]);
  }
  return answer;
}

static inline void
train_net(RecurNN *net, float *features, float *target){
  bptt_advance(net);
  float *answer = tanh_opinion(net, features);

  /*Tanh derivative is allegedly 1 - y * y
    I believe this is meant to happen here. */
  for (int i = 0; i < net->output_size; i++){
    float a = answer[i];
    net->bptt->o_error[i] = (1 - a * a) * (target[i] - a);
  }
  bptt_calc_deltas(net);
}

static inline void
consolidate_and_apply_learning(GstParrot *self)
{
  RecurNN *net = self->training_nets[0];
  if (REGULAR_PGM_DUMP)
    rnn_multi_pgm_dump(net, "ihw hhw");
  else if (PERIODIC_PGM_DUMP && net->generation % PERIODIC_PGM_DUMP == 0)
    rnn_multi_pgm_dump(net, "how ihw hod ihd hom ihm");
  bptt_consolidate_many_nets(self->training_nets, self->n_channels);
  rnn_condition_net(net);
  possibly_save_net(self->net, self->net_filename);
}

static inline void
maybe_learn(GstParrot *self){
  int i, j, k;
  int len_i = self->incoming_end - self->incoming_start;
  if (len_i < 0)
    len_i += self->queue_size;
  int half_window = PARROT_WINDOW_SIZE / 2;
  int chunk_size =  half_window * self->n_channels;
  const float *window = self->window;

  while (len_i >= chunk_size){
    s16 *buffer_i = self->incoming_queue + self->incoming_start;
    for (j = 0; j < self->n_channels; j++){
      ParrotChannel *c = & self->channels[j];
      float *target = c->mdct_target;

      /* Situation from previous round:

                 | side 1  |  side 2  |
        pcm_now  | -1      |   -2     |
        pcm_prev | -2      |   -1     | ready

        pcm_prev should predict pcm_now
       */
      pcm_to_features(self->mfcc_factory, c->features, c->pcm_prev);

      /*load first half of pcm_prev, second part of pcm_now.*/
      /*second part of pcm_now retains previous data */
      /*NB: copy into pcm_{prev,now} casts to float*/
      for(i = 0, k = j; i < half_window; i++, k += self->n_channels){
        c->pcm_prev[i] = buffer_i[k] * window[i];
        c->pcm_now[half_window + i] = buffer_i[k] * window[half_window + i];
      }

      /*
                 | side 1  |  side 2  |
        pcm_now  | -1      |   0      | ready
        pcm_prev |  0      |  -1      |

       */
      mdct_forward(&self->mdct_lut, c->pcm_now, target);
      train_net(c->train_net, c->features, target);

      float *tmp;
      tmp = c->pcm_now;
      c->pcm_now = c->pcm_prev;
      c->pcm_prev = tmp;
    }

    consolidate_and_apply_learning(self);
    RecurNN *net = self->channels[0].train_net;
    rnn_log_net(net);

    self->incoming_start += chunk_size;
    self->incoming_start %= self->queue_size;
    len_i -= chunk_size;
  }
}

static inline void
generate_audio(GstParrot *self){
  int i, j, k;
  int len_o = self->outgoing_end - self->outgoing_start;
  if (len_o < 0)
    len_o += self->queue_size;
  int half_window = PARROT_WINDOW_SIZE / 2;
  int chunk_size =  half_window * self->n_channels;
  const float *window = self->window;

  while (len_o < self->queue_size){
    s16 *buffer_o = self->outgoing_queue + self->outgoing_end;
    for (j = 0; j < self->n_channels; j++){
      ParrotChannel *c = & self->channels[j];

      pcm_to_features(self->mfcc_factory, c->features, c->play_prev);

      float *answer = tanh_opinion(c->dream_net, c->features);
      mdct_backward(&self->mdct_lut, answer, c->play_now);

      for(i = 0, k = j; i < half_window; i++, k += self->n_channels){
        float s = (c->play_prev[half_window + i] * window[half_window - 1 - i] +
            c->play_now[i] * window[i]);
        /*window is scaled by 1 / 32768; scale back, doubly */
        buffer_o[k] = s * 1073741823.99f;
      }
      float *tmp;
      tmp = c->play_now;
      c->play_now = c->play_prev;
      c->play_prev = tmp;
    }
    self->outgoing_end += chunk_size;
    if (self->outgoing_end > self->queue_size)
      self->outgoing_end -= self->queue_size;
    len_o += chunk_size;
  }
}




static GstFlowReturn
gst_parrot_transform (GstBaseTransform * base, GstBuffer *inbuf, GstBuffer *outbuf)
{
  GstParrot *self = GST_PARROT(base);
  GstFlowReturn ret = GST_FLOW_OK;

  queue_audio_segment(self, inbuf);
  if (self->training)
    maybe_learn(self);
  if (self->playing)
    generate_audio(self);
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
