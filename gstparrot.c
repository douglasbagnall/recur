/* Copyright 2013 Douglas Bagnall <douglas@halo.gen.nz> LGPL */
#include "gstparrot.h"
#include "audio-common.h"
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
  PROP_TRAINING,
  PROP_PLAYING,
};

#define DEFAULT_PROP_PGM_DUMP ""
#define DEFAULT_PROP_LOG_FILE ""
#define DEFAULT_PROP_SAVE_NET NULL
#define DEFAULT_PROP_FORGET 0
#define DEFAULT_PROP_PLAYING 1
#define DEFAULT_PROP_TRAINING 1
#define DEFAULT_HIDDEN_SIZE 199
#define DEFAULT_LEARN_RATE 0.0001
#define MIN_HIDDEN_SIZE 1
#define MAX_HIDDEN_SIZE 1000000
#define LEARN_RATE_MIN 0.0
#define LEARN_RATE_MAX 1.0

/* static_functions */
static void gst_parrot_class_init(GstParrotClass *g_class);
static void gst_parrot_init(GstParrot *self);
static void gst_parrot_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_parrot_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static GstFlowReturn gst_parrot_transform_ip(GstBaseTransform *base, GstBuffer *buf);
static gboolean gst_parrot_setup(GstAudioFilter * filter, const GstAudioInfo * info);
static void maybe_start_logging(GstParrot *self);


#define gst_parrot_parent_class parent_class
G_DEFINE_TYPE (GstParrot, gst_parrot, GST_TYPE_AUDIO_FILTER)

#define PARROT_CAPS_STRING "audio/x-raw, format = (string) " QUOTE(PARROT_FORMAT) \
  ", rate = (int) " QUOTE(PARROT_RATE)                                  \
  ", channels = (int) [ " QUOTE(PARROT_MIN_CHANNELS) " , " QUOTE(PARROT_MAX_CHANNELS)  \
  " ] , layout = (string) interleaved"/*", channel-mask = (bitmask)0x0"*/

static inline void
init_channel(ParrotChannel *c, RecurNN *train_net, int id, float learn_rate)
{
  u32 dream_flags = train_net->flags & ~(RNN_NET_FLAG_OWN_WEIGHTS | \
      RNN_NET_FLAG_OWN_BPTT);
  c->train_net = train_net;
  c->dream_net = rnn_clone(train_net, dream_flags, RECUR_RNG_SUBSEED, NULL);
  c->pcm_now = zalloc_aligned_or_die(PARROT_WINDOW_SIZE * sizeof(float));
  c->pcm_prev = zalloc_aligned_or_die(PARROT_WINDOW_SIZE * sizeof(float));
  c->play_now = zalloc_aligned_or_die(PARROT_WINDOW_SIZE * sizeof(float));
  c->play_prev = zalloc_aligned_or_die(PARROT_WINDOW_SIZE * sizeof(float));
  c->mdct_now = zalloc_aligned_or_die(PARROT_WINDOW_SIZE / 2 * sizeof(float));
  c->mdct_prev = zalloc_aligned_or_die(PARROT_WINDOW_SIZE / 2 * sizeof(float));
  if (PGM_DUMP_FEATURES){
    c->mfcc_image = temporal_ppm_alloc(PARROT_N_FEATURES, 300, "parrot-mfcc", id,
        PGM_DUMP_COLOUR, NULL);
    c->pcm_image = temporal_ppm_alloc(PARROT_WINDOW_SIZE, 300, "parrot-pcm", id,
        PGM_DUMP_COLOUR, NULL);
    c->pcm_image2 = temporal_ppm_alloc(PARROT_WINDOW_SIZE, 300, "parrot-pcm2", id,
        PGM_DUMP_COLOUR, NULL);
    c->dct_image = temporal_ppm_alloc(PARROT_WINDOW_SIZE / 2, 300, "parrot-dct", id,
        PGM_DUMP_COLOUR, NULL);
    c->answer_image = temporal_ppm_alloc(PARROT_WINDOW_SIZE / 2, 300, "parrot-out", id,
        PGM_DUMP_COLOUR, NULL);
  }
  else {
    c->mfcc_image = NULL;
    c->pcm_image = NULL;
    c->pcm_image2 = NULL;
    c->dct_image = NULL;
    c->answer_image = NULL;
  }
}

static inline void
finalise_channel(ParrotChannel *c)
{
  rnn_delete_net(c->dream_net);
  free(c->pcm_prev);
  free(c->pcm_now);
  free(c->play_prev);
  free(c->play_now);
  free(c->mdct_prev);
  free(c->mdct_now);
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
    rnn_save_net(self->net, self->net_filename, 1);
  }
  if (self->mfcc_factory){
    recur_audio_binner_delete(self->mfcc_factory);
  }
  if (self->channels){
    for (int i = 0; i < self->n_channels; i++){
      finalise_channel(&self->channels[i]);
    }
    free(self->channels);
    self->channels = NULL;
    rnn_delete_training_set(self->training_nets, self->n_channels, 0);
  }
  else if (self->net){
    rnn_delete_net(self->net);
  }
  self->net = NULL;

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

  g_object_class_install_property (gobject_class, PROP_PLAYING,
      g_param_spec_boolean("playing", "playing",
          "Construct imaginary audio to play",
          DEFAULT_PROP_PLAYING,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_TRAINING,
      g_param_spec_boolean("training", "training",
          "Learn from incoming audio",
          DEFAULT_PROP_TRAINING,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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


  trans_class->transform_ip = GST_DEBUG_FUNCPTR (gst_parrot_transform_ip);
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
  self->training = DEFAULT_PROP_TRAINING;
  self->playing = DEFAULT_PROP_PLAYING;
  GST_INFO("gst parrot init\n");
}

static void
reset_net_filename(GstParrot *self){
  char s[200];
  snprintf(s, sizeof(s), "parrot-i%d-h%d-o%d-%dHz.net",
      PARROT_N_FEATURES, self->hidden_size, PARROT_WINDOW_SIZE,
      PARROT_RATE);
  if (self->net_filename){
    free(self->net_filename);
  }
  self->net_filename = strdup(s);
}

static RecurNN *
load_or_create_net(GstParrot *self){
  reset_net_filename(self);
  RecurNN *net = TRY_RELOAD ? rnn_load_net(self->net_filename) : NULL;
  if (net){
    if (net->output_size != PARROT_WINDOW_SIZE / 2){
      GST_WARNING("loaded net doesn't seem to match!");
      rnn_delete_net(net);
      net = NULL;
    }
  }
  if (net == NULL){
    net = rnn_new(PARROT_N_FEATURES, self->hidden_size,
        PARROT_N_FEATURES, PARROT_RNN_FLAGS, PARROT_RNG_SEED,
        NULL, PARROT_BPTT_DEPTH, self->learn_rate, MOMENTUM,
        PARROT_PRESYNAPTIC_NOISE, RNN_RELU);
    rnn_randomise_weights_auto(net);
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
    int chunk_size = info->channels * PARROT_WINDOW_SIZE;
    self->queue_size = chunk_size * PARROT_QUEUE_N_CHUNKS;
    self->incoming_queue = malloc_aligned_or_die(self->queue_size * sizeof(s16));
    self->outgoing_queue = malloc_aligned_or_die(self->queue_size * sizeof(s16));
  }
  self->window = malloc_aligned_or_die(PARROT_WINDOW_SIZE * sizeof(float));
  recur_window_init(self->window, PARROT_WINDOW_SIZE,
      RECUR_WINDOW_VORBIS, 1.0f / 32768.0f);

  mdct_init(&self->mdct_lut, PARROT_WINDOW_SIZE);

  if (self->mfcc_factory == NULL){
    self->mfcc_factory = recur_audio_binner_new(PARROT_WINDOW_SIZE,
        RECUR_WINDOW_NONE,
        PARROT_N_FFT_BINS,
        PARROT_MFCC_MIN_FREQ,
        PARROT_MFCC_MAX_FREQ,
        PARROT_MFCC_KNEE_FREQ,
        PARROT_MFCC_FOCUS_FREQ,
        PARROT_RATE,
        1.0f,
        PARROT_VALUE_SIZE
    );
  }
  if (self->net == NULL){
    self->net = load_or_create_net(self);
  }
  if (self->training_nets == NULL){
    self->training_nets = rnn_new_training_set(self->net, self->n_channels);
  }
  if (self->channels == NULL){
    self->channels = malloc_aligned_or_die(self->n_channels * sizeof(ParrotChannel));
    for (int i = 0; i < self->n_channels; i++){
      init_channel(&self->channels[i], self->training_nets[i], i, self->learn_rate);
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
      rnn_multi_pgm_dump(self->net, strvalue, "parrot");
      break;

    case PROP_SAVE_NET:
      strvalue = g_value_get_string(value);
      if (strvalue && strvalue[0] != 0){
        rnn_save_net(self->net, strvalue, 1);
      }
      else {
        rnn_save_net(self->net, self->net_filename, 1);
      }
      break;

    case PROP_LOG_FILE:
      /*defer setting the actual log file, in case the nets aren't ready yet*/
      if (self->pending_logfile){
        free(self->pending_logfile);
      }
      self->pending_logfile = g_value_dup_string(value);
      GST_DEBUG("set log file: %s", self->pending_logfile);
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

    case PROP_PLAYING:
      self->playing = g_value_get_boolean(value);
      break;

    case PROP_TRAINING:
      self->training = g_value_get_boolean(value);
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

  case PROP_PLAYING:
    g_value_set_boolean(value, self->playing);
    break;

  case PROP_TRAINING:
    g_value_set_boolean(value, self->training);
    break;

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



static inline void
possibly_save_net(RecurNN *net, char *filename)
{
  GST_LOG("possibly saving to %s", filename);
  if (PERIODIC_SAVE_NET && (net->generation & 511) == 0){
    rnn_save_net(net, filename, 1);
  }
}

static inline float *
tanh_opinion(RecurNN *net, float *in){
  float *answer = rnn_opinion(net, in, 0);
  for (int i = 0; i < net->output_size; i++){
    answer[i] = fast_tanhf(answer[i]);
  }
  return answer;
}

static inline float *
train_net(RecurNN *net, float *features, float *target){
  rnn_bptt_advance(net);
  float *answer = tanh_opinion(net, features);

  /*Tanh derivative is allegedly 1 - y * y
    I believe this is meant to happen here. */
  for (int i = 0; i < net->output_size; i++){
    float a = answer[i];
    net->bptt->o_error[i] = (1 - a * a) * (target[i] - a);
  }
  rnn_bptt_calc_deltas(net, 0);
  return answer;
}

static inline void
maybe_add_ppm_row(TemporalPPM *ppm, float *row, int yes_do_it)
{
  if (ppm && yes_do_it){
    temporal_ppm_add_row(ppm, row);
  }
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
      /* Situation from previous round:

                 | side 1  |  side 2  |
        pcm_now  | -1      |   -2     |
        pcm_prev | -2      |   -1     | ready

        mdct_prev is based on pcm_prev.
        mdct_prev should predict mdct_now
       */
      /*load first half of pcm_prev, second part of pcm_now.*/
      /*second part of pcm_now retains previous data */
      /*NB: copy into pcm_{prev,now} casts to float*/
      for(i = 0, k = j; i < half_window; i++, k += self->n_channels){
        c->pcm_prev[i] = buffer_i[k] * window[i];
        c->pcm_now[half_window + i] = buffer_i[k] * window[half_window + i];
      }
      //maybe_add_ppm_row(c->pcm_image, c->pcm_now, PGM_DUMP_LEARN);
      /*
                 | side 1  |  side 2  |
        pcm_now  | -1      |   0      | ready
        pcm_prev |  0      |  -1      |

       */
      mdct_forward(&self->mdct_lut, c->pcm_now, c->mdct_now);
      maybe_add_ppm_row(c->dct_image, c->mdct_now, PGM_DUMP_LEARN);

      float *answer = train_net(c->train_net, c->mdct_prev, c->mdct_now);
      maybe_add_ppm_row(c->answer_image, answer, PGM_DUMP_LEARN);

      float *tmp;
      tmp = c->pcm_now;
      c->pcm_now = c->pcm_prev;
      c->pcm_prev = tmp;

      tmp = c->mdct_prev;
      c->mdct_prev = c->mdct_now;
      c->mdct_now = tmp;
    }

    RecurNN *net = self->training_nets[0];
    if (PERIODIC_PGM_DUMP && net->generation % PERIODIC_PGM_DUMP == 0){
      rnn_multi_pgm_dump(net, "how ihw", "parrot");
    }
    rnn_apply_learning(self->net, RNN_MOMENTUM_WEIGHTED, self->net->bptt->momentum);
    rnn_condition_net(self->net);
    self->net->generation = net->generation;
    possibly_save_net(self->net, self->net_filename);
    rnn_log_net(net);

    self->incoming_start += chunk_size;
    self->incoming_start %= self->queue_size;
    len_i -= chunk_size;
  }
}

static inline void
fill_audio_chunk(GstParrot *self, s16 *dest){
  int i, j;
  int half_window = PARROT_WINDOW_SIZE / 2;
  int n_channels = self->n_channels;
  const float *window = self->window;
  for (j = 0; j < n_channels; j++){
    ParrotChannel *c = & self->channels[j];
    float *answer = c->dream_net->output_layer;
    maybe_add_ppm_row(c->pcm_image, c->play_prev, PGM_DUMP_OUT);
    answer = tanh_opinion(c->dream_net, answer);
    maybe_add_ppm_row(c->answer_image, answer, PGM_DUMP_OUT);
    mdct_backward(&self->mdct_lut, answer, c->play_now);
    for(i = 0; i < half_window; i++){
      float s = (c->play_prev[half_window + i] * window[half_window + i] +
          c->play_now[i] * window[i]);
      //s = (c->play_prev[half_window + i] + c->play_now[i])/ 32768;
      /*window is scaled by 1 / 32768; scale back, doubly */
      //DEBUG("k is %d. len_o %d", k, len_o);
      dest[j + i * n_channels] = s * 32768 * 32768;
      answer[i] *= 1.0f + cheap_gaussian_noise(&self->net->rng);
    }
    float *tmp;
    tmp = c->play_now;
    c->play_now = c->play_prev;
    c->play_prev = tmp;
  }
}

static inline void
fill_audio_segment(GstParrot *self, GstBuffer *outbuf)
{
  /*calculate enough audio in the audio queue, then copy it across. It is
    tricky to do directly because of the overlapping windows. */
  GstMapInfo map;
  gst_buffer_map(outbuf, &map, GST_MAP_WRITE);
  int len16 = map.size / sizeof(s16);
  int qlen = self->outgoing_end - self->outgoing_start;
  if (qlen < 0){
    qlen += self->queue_size;
  }

  int half_window = PARROT_WINDOW_SIZE / 2;
  int chunk_size =  half_window * self->n_channels;

  while (qlen < len16 + half_window){
    GST_LOG("filling chunk at %d / %d", self->outgoing_end, self->queue_size);
    fill_audio_chunk(self, self->outgoing_queue + self->outgoing_end);
    self->outgoing_end += chunk_size;
    qlen += chunk_size;
    if (self->outgoing_end >= self->queue_size){
      self->outgoing_end = 0;
    }
  }

  int n_samples = MIN(len16, self->queue_size - self->outgoing_start);
  GST_LOG("copying buffer of %d samples to buffer of %d",
      n_samples, len16);

  memcpy(map.data, self->outgoing_queue + self->outgoing_start,
      n_samples * sizeof(s16));

  if (n_samples < len16){
    int remainder = len16 - n_samples;
    GST_LOG("copying remainder of %d samples to buffer of %d",
        remainder, len16);
    memcpy(map.data + n_samples * sizeof(s16),
        self->outgoing_queue, remainder * sizeof(s16));
    self->outgoing_start = remainder;
  }
  else {
    self->outgoing_start += n_samples;
  }
  gst_buffer_unmap(outbuf, &map);
}




static GstFlowReturn
gst_parrot_transform_ip(GstBaseTransform * base, GstBuffer *buf)
{
  GstParrot *self = GST_PARROT(base);
  GstFlowReturn ret = GST_FLOW_OK;
  if (self->training){
    queue_audio_segment(buf, self->incoming_queue, self->queue_size,
        &self->incoming_start, &self->incoming_end);
    maybe_learn(self);
  }
  if (self->playing){
    fill_audio_segment(self, buf);
  }
  return ret;
}


static gboolean
plugin_init (GstPlugin * plugin)
{
  gboolean parrot = gst_element_register (plugin, "parrot", GST_RANK_NONE,\
      GST_TYPE_PARROT);
  return parrot;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    parrot,
    "Parrot audio streams",
    plugin_init, VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
