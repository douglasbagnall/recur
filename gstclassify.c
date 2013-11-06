/* GStreamer
 * Copyright (C) <2013> Douglas Bagnall <douglas@halo.gen.nz>
 *
 */

#include "gstclassify.h"
#include "audio-common.h"
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
  PROP_CLASSES,
  PROP_FORGET,
  PROP_LEARN_RATE,
  PROP_HIDDEN_SIZE,
  PROP_MOMENTUM,
  PROP_MOMENTUM_SOFT_START,
  PROP_MFCCS,
  PROP_SAVE_NET,
  PROP_PGM_DUMP,
  PROP_LOG_FILE,
};

#define DEFAULT_PROP_TARGET ""
#define DEFAULT_PROP_PGM_DUMP ""
#define DEFAULT_PROP_LOG_FILE ""
#define DEFAULT_PROP_SAVE_NET NULL
#define DEFAULT_PROP_MFCCS 0
#define DEFAULT_PROP_MOMENTUM 0.95f
#define DEFAULT_PROP_MOMENTUM_SOFT_START 0.0f
#define DEFAULT_PROP_CLASSES 2
#define DEFAULT_PROP_FORGET 0
#define DEFAULT_HIDDEN_SIZE 199
#define DEFAULT_LEARN_RATE 0.0001
#define MIN_PROP_CLASSES 1
#define MAX_PROP_CLASSES 1000000
#define MIN_HIDDEN_SIZE 1
#define MAX_HIDDEN_SIZE 1000000
#define LEARN_RATE_MIN 0.0
#define LEARN_RATE_MAX 1.0
#define MOMENTUM_MIN 0.0
#define MOMENTUM_MAX 1.0
#define MIN_PROP_MFCCS 0
#define MAX_PROP_MFCCS (CLASSIFY_N_FFT_BINS - 1)
#define MOMENTUM_SOFT_START_MAX 1e9
#define MOMENTUM_SOFT_START_MIN 0

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
static void maybe_parse_target_string(GstClassify *self);
static void maybe_start_logging(GstClassify *self);


#define gst_classify_parent_class parent_class
G_DEFINE_TYPE (GstClassify, gst_classify, GST_TYPE_AUDIO_FILTER);

#define CLASSIFY_CAPS_STRING "audio/x-raw, format = (string) " QUOTE(CLASSIFY_FORMAT) \
      ", rate = (int) " QUOTE(CLASSIFY_RATE) \
      ", channels = (int) [ " QUOTE(CLASSIFY_MIN_CHANNELS) " , " QUOTE(CLASSIFY_MAX_CHANNELS) " ] " \
      ", layout = (string) interleaved, channel-mask = (bitmask)0x0"


static inline void
init_channel(ClassifyChannel *c, RecurNN *net, int id, float learn_rate)
{
  u32 flags = net->flags & ~RNN_NET_FLAG_OWN_WEIGHTS;
  c->net = rnn_clone(net, flags, RECUR_RNG_SUBSEED, NULL);
  c->net->bptt->learn_rate = learn_rate;
  c->pcm_next = zalloc_aligned_or_die(CLASSIFY_WINDOW_SIZE * sizeof(float));
  c->pcm_now = zalloc_aligned_or_die(CLASSIFY_WINDOW_SIZE * sizeof(float));
  c->features = zalloc_aligned_or_die(net->input_size * sizeof(float));
  if (PGM_DUMP_FEATURES){
    c->mfcc_image = temporal_ppm_alloc(net->input_size, 300, "mfcc", id,
        PGM_DUMP_COLOUR);
  }
  else {
    c->mfcc_image = NULL;
  }
}

static inline void
finalise_channel(ClassifyChannel *c)
{
  rnn_delete_net(c->net);
  free(c->pcm_next);
  free(c->pcm_now);
  free(c->features);
  if (c->mfcc_image){
    temporal_ppm_free(c->mfcc_image);
    c->mfcc_image = NULL;
  }
}

/* Clean up */
static void
gst_classify_finalize (GObject * obj){
  GST_DEBUG("in gst_classify_finalize!\n");
  GstClassify *self = GST_CLASSIFY(obj);
  rnn_save_net(self->net, self->net_filename);
  recur_audio_binner_delete(self->mfcc_factory);
  if (self->channels){
    for (int i = 0; i < self->n_channels; i++){
      finalise_channel(&self->channels[i]);
      self->subnets[i] = NULL;
    }
    free(self->channels);
    self->channels = NULL;
    free(self->subnets);
  }
  free(self->incoming_queue);
  rnn_delete_net(self->net);
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
      "Audio classifying element",
      "Analyzer/Audio",
      "Classifies audio",
      "Douglas Bagnall <douglas@halo.gen.nz>");

  g_object_class_install_property (gobject_class, PROP_TARGET,
      g_param_spec_string("target", "target",
          "Target outputs for all channels (dot separated)",
          DEFAULT_PROP_TARGET,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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

  g_object_class_install_property (gobject_class, PROP_CLASSES,
      g_param_spec_int("classes", "classes",
          "Use this many classes",
          MIN_PROP_CLASSES, MAX_PROP_CLASSES,
          DEFAULT_PROP_CLASSES,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MFCCS,
      g_param_spec_int("mfccs", "mfccs",
          "Use this many MFCCs, or zero for fft bins",
          MIN_PROP_MFCCS, MAX_PROP_MFCCS,
          DEFAULT_PROP_MFCCS,
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

  g_object_class_install_property (gobject_class, PROP_MOMENTUM_SOFT_START,
      g_param_spec_float("momentum-soft-start", "momentum-soft-start",
          "Ease into momentum over many generations",
          MOMENTUM_SOFT_START_MIN, MOMENTUM_SOFT_START_MAX,
          DEFAULT_PROP_MOMENTUM_SOFT_START,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MOMENTUM,
      g_param_spec_float("momentum", "momentum",
          "(eventual) momentum",
          MOMENTUM_MIN, MOMENTUM_MAX,
          DEFAULT_PROP_MOMENTUM,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_HIDDEN_SIZE,
      g_param_spec_int("hidden-size", "hidden-size",
          "Size of the RNN hidden layer",
          MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE,
          DEFAULT_HIDDEN_SIZE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));


  trans_class->transform_ip = GST_DEBUG_FUNCPTR (gst_classify_transform_ip);
  af_class->setup = GST_DEBUG_FUNCPTR (gst_classify_setup);
  GST_INFO("gst audio class init\n");
}

static void
gst_classify_init (GstClassify * self)
{
  self->channels = NULL;
  self->target_string = NULL;
  self->n_channels = 0;
  self->mfcc_factory = NULL;
  self->incoming_queue = NULL;
  self->net_filename = NULL;
  self->pending_logfile = NULL;
  self->learn_rate = DEFAULT_LEARN_RATE;
  self->hidden_size = DEFAULT_HIDDEN_SIZE;
  self->momentum_soft_start = DEFAULT_PROP_MOMENTUM_SOFT_START;
  self->momentum = DEFAULT_PROP_MOMENTUM;
  GST_INFO("gst classify init\n");
}


static void
reset_net_filename(GstClassify *self){
  char s[200];
  int n_features = self->mfccs ? self->mfccs : CLASSIFY_N_FFT_BINS;
  snprintf(s, sizeof(s), "classify-i%d-h%d-o%d-b%d-%dHz-w%d.net",
      n_features, self->hidden_size, self->n_classes,
      CLASSIFY_BIAS, CLASSIFY_RATE, CLASSIFY_WINDOW_SIZE);
  if (self->net_filename){
    free(self->net_filename);
  }
  self->net_filename = strdup(s);
}


static RecurNN *
load_or_create_net(GstClassify *self){
  reset_net_filename(self);
  RecurNN *net = TRY_RELOAD ? rnn_load_net(self->net_filename) : NULL;
  if (net){
    if (net->output_size != self->n_classes){
      GST_WARNING("loaded net doesn't seem to match!");
      rnn_delete_net(net);
      net = NULL;
    }
  }
  if (net == NULL){
    int n_features = self->mfccs ? self->mfccs : CLASSIFY_N_FFT_BINS;
    net = rnn_new(n_features, self->hidden_size,
        self->n_classes, CLASSIFY_RNN_FLAGS, CLASSIFY_RNG_SEED,
        NULL, CLASSIFY_BPTT_DEPTH, self->learn_rate, self->momentum, MOMENTUM_WEIGHT,
        CLASSIFY_BATCH_SIZE);
  }
  else {
    rnn_set_log_file(net, NULL, 0);
  }
  return net;
}


static gboolean
gst_classify_setup(GstAudioFilter *base, const GstAudioInfo *info){
  GST_INFO("gst_classify_setup\n");
  GstClassify *self = GST_CLASSIFY(base);
  //self->info = info;
  self->n_channels = info->channels;
  if (self->incoming_queue == NULL){
    self->queue_size = info->channels * CLASSIFY_QUEUE_PER_CHANNEL;
    self->incoming_queue = malloc_aligned_or_die(self->queue_size * sizeof(s16));
  }

  if (self->mfcc_factory == NULL){
    self->mfcc_factory = recur_audio_binner_new(CLASSIFY_WINDOW_SIZE,
        RECUR_WINDOW_HANN,
        CLASSIFY_N_FFT_BINS,
        CLASSIFY_MFCC_MIN_FREQ,
        CLASSIFY_MFCC_MAX_FREQ,
        CLASSIFY_RATE,
        1.0f / 32768,
        CLASSIFY_VALUE_SIZE
    );
  }
  if (self->net == NULL){
    self->net = load_or_create_net(self);
  }
  if (self->channels == NULL){
    self->channels = malloc_aligned_or_die(self->n_channels * sizeof(ClassifyChannel));
    self->subnets = malloc_aligned_or_die(self->n_channels * sizeof(RecurNN *));
    for (int i = 0; i < self->n_channels; i++){
      init_channel(&self->channels[i], self->net, i, self->learn_rate);
      self->subnets[i] = self->channels[i].net;
    }
  }
  maybe_start_logging(self);
  maybe_parse_target_string(self);

  GST_DEBUG_OBJECT (self,
      "info: %" GST_PTR_FORMAT, info);
  DEBUG("found %d channels", self->n_channels);
  return TRUE;
}

static inline void
maybe_parse_target_string(GstClassify *self){
  int i;
  if (self->target_string == NULL || self->channels == NULL){
    GST_DEBUG("not parsing NULL target string (%p) channels is %p",
        self->target_string, self->channels);
    return;
  }
  char *s = self->target_string;
  if (*s == 0){
    for (i = 0; i < self->n_channels; i++){
      self->channels[i].current_target = -1;
    }
    self->training = 0;
    goto cleanup;
  }
  char *e;
  long x;
  for (i = 0; i < self->n_channels; i++){
    x = strtol(s, &e, 10);
    GST_DEBUG("channel %d got %ld s is %s, %p  e is %p", i, x, s, s, e);
    if (s == e){ /*no digits at all */
      goto parse_error;
    }
    self->channels[i].current_target = x;
    s = e + 1;
    if (*e == '\0'  && i != self->n_channels - 1){ /*ran out of numbers */
      goto parse_error;
    }
  }
  GST_DEBUG("target %d", self->channels[0].current_target);
  self->training = 1;
  goto cleanup;
 parse_error:
  GST_DEBUG("Can't parse '%s' into %d channels: stopping after %d",
      self->target_string, self->n_channels, i);
 cleanup:
  free(self->target_string);
  self->target_string = NULL;
  return;
}

static void
maybe_start_logging(GstClassify *self){
  if (self->pending_logfile && self->subnets){
    if (self->pending_logfile[0] == 0){
      rnn_set_log_file(self->subnets[0], NULL, 0);
    }
    else {
      rnn_set_log_file(self->subnets[0], self->pending_logfile, 1);
    }
    free(self->pending_logfile);
    self->pending_logfile = NULL;
  }
}


static void
gst_classify_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstClassify *self = GST_CLASSIFY (object);
  GST_DEBUG("gst_classify_set_property\n");
  if (value){
    const char *strvalue;
    switch (prop_id) {
    case PROP_TARGET:
      if (self->target_string){
        free(self->target_string);
      }
      self->target_string = g_value_dup_string(value);
      maybe_parse_target_string(self);
      break;

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
          rnn_forget_history(self->channels[i].net, bptt_too);
        }
      }
      break;

    case PROP_LEARN_RATE:
      self->learn_rate = g_value_get_float(value);
      if (self->net){
        float learn_rate = g_value_get_float(value);
        for (int i = 0; i < self->n_channels; i++){
          self->channels[i].net->bptt->learn_rate = learn_rate;
        }
      }
      break;

    case PROP_MOMENTUM_SOFT_START:
      self->momentum_soft_start = g_value_get_float(value);
      break;

    case PROP_MOMENTUM:
      self->momentum = g_value_get_float(value);
      break;

      /*CLASSES, MFCCS, and HIDDEN_SIZE have no effect if set late (after net creation)
       */
#define SET_INT_IF_NOT_TOO_LATE(attr, name) do {                        \
        if (self->net == NULL){                                         \
          self->attr = g_value_get_int(value);                          \
        }                                                               \
        else {                                                          \
          GST_WARNING("it is TOO LATE to set " name                     \
              " (is %d, requested %d)", self->attr,                     \
              g_value_get_int(value));                                  \
        }} while (0)

    case PROP_MFCCS:
      SET_INT_IF_NOT_TOO_LATE(mfccs, "number of MFCCs");
      break;

    case PROP_CLASSES:
      SET_INT_IF_NOT_TOO_LATE(n_classes, "number of classes");
      break;

    case PROP_HIDDEN_SIZE:
      SET_INT_IF_NOT_TOO_LATE(hidden_size, "hidden layer size");
      break;

#undef SET_INT_IF_NOT_TOO_LATE

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
  case PROP_CLASSES:
    g_value_set_int(value, self->n_classes);
    break;
  case PROP_MFCCS:
    g_value_set_int(value, self->mfccs);
    break;
  case PROP_LEARN_RATE:
    g_value_set_float(value, self->learn_rate);
    break;
  case PROP_MOMENTUM:
    g_value_set_float(value, self->momentum);
    break;
  case PROP_MOMENTUM_SOFT_START:
    g_value_set_float(value, self->momentum_soft_start);
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
    rnn_save_net(net, filename);
  }
}


static inline void
send_message(GstClassify *self, float mean_err)
{
  GstMessage *msg;
  char *name = "classify";
  GstStructure *s;
  s = gst_structure_new (name,
      "error", G_TYPE_FLOAT, mean_err,
      NULL);
  for (int i = 0; i < self->n_channels; i++){
    char key[50];
    RecurNN *net = self->channels[i].net;
    for (int j = 0; j < net->output_size; j++){
      snprintf(key, sizeof(key), "channel %d, output %d", i, j);
      gst_structure_set(s, key, G_TYPE_FLOAT, net->bptt->o_error[j], NULL);
    }
    snprintf(key, sizeof(key), "channel %d winner", i);
    gst_structure_set(s, key, G_TYPE_INT, self->channels[i].current_winner, NULL);
  }
  msg = gst_message_new_element(GST_OBJECT(self), s);
  gst_element_post_message(GST_ELEMENT(self), msg);
}


static inline void
pcm_to_features(RecurAudioBinner *mf, float *features, float *pcm, int mfccs){
  float *answer;
  int n_features;
  if (mfccs){
    answer = recur_extract_mfccs(mf, pcm) + 1;
    n_features = mfccs;
  }
  else {
    answer = recur_extract_log_freq_bins(mf, pcm);
    n_features = CLASSIFY_N_FFT_BINS;
  }
  for (int i = 0; i < n_features; i++){
    features[i] = answer[i];
  }
}

static inline float
train_channel(ClassifyChannel *c){
  RecurNN *net = c->net;
  bptt_advance(net);
  float *answer = rnn_opinion(net, c->features);
  c->current_winner = softmax_best_guess(net->bptt->o_error, answer,
      net->output_size);
  net->bptt->o_error[c->current_target] += 1.0f;
  bptt_calc_deltas(net);
  return net->bptt->o_error[c->current_target];
}


static inline void
maybe_learn(GstClassify *self){
  int i, j, k;
  int len = self->incoming_end - self->incoming_start;
  if (len < 0){ //XXX or less than or equal?
    len += self->queue_size;
  }
  int chunk_size = CLASSIFY_HALF_WINDOW * self->n_channels;

  while (len >= chunk_size){
    float err_sum = 0.0f;
    float winners = 0.0f;
    s16 *buffer_i = self->incoming_queue + self->incoming_start;

    for (j = 0; j < self->n_channels; j++){
      /*load first half of pcm_next, second part of pcm_now.*/
      /*second part of pcm_next retains previous data */
      ClassifyChannel *c = &self->channels[j];
      RecurNN *net = c->net;
      for(i = 0, k = j; i < CLASSIFY_HALF_WINDOW; i++, k += self->n_channels){
        c->pcm_next[i] = buffer_i[k];
        c->pcm_now[CLASSIFY_HALF_WINDOW + i] = buffer_i[k];
      }

      /*get the features -- after which pcm_now is finished with. */
      pcm_to_features(self->mfcc_factory, c->features, c->pcm_now, self->mfccs);
      if (c->mfcc_image){
        temporal_ppm_add_row(c->mfcc_image, c->features);
        //temporal_ppm_add_row(c->mfcc_image, c->pcm_now);
      }
      float *answer;

      int valid_target = c->current_target >= 0 && c->current_target < self->n_classes;
      int training = valid_target && self->training;

      //GST_DEBUG("channel %d target %d", j , c->current_target);
      if (training){
        err_sum += train_channel(c);
        winners += c->current_winner == c->current_target;
      }
      else{
        answer = rnn_opinion(net, c->features);
        c->current_winner = softmax_best_guess(net->bptt->o_error, answer,
            net->output_size);
        if (valid_target){
          err_sum += net->bptt->o_error[c->current_target];
        }
      }

      float *tmp;
      tmp = c->pcm_next;
      c->pcm_next = c->pcm_now;
      c->pcm_now = tmp;
    }

    if (self->training){
      RecurNN *net = self->subnets[0];
      if (PERIODIC_PGM_DUMP && net->generation % PERIODIC_PGM_DUMP == 0){
        rnn_multi_pgm_dump(net, "how ihw hod ihd hom ihm");
      }
      if (self->momentum_soft_start){
        float x = self->momentum_soft_start;
        net->bptt->momentum = MIN(1.0f - x / (1 + net->generation + 2 * x), self->momentum);
        if (net->bptt->momentum == self->momentum){
          self->momentum_soft_start = 0;
        }
      }

      bptt_consolidate_many_nets(self->subnets, self->n_channels, 1);
      rnn_condition_net(self->net);
      possibly_save_net(self->net, self->net_filename);
      rnn_log_net(net);
      bptt_log_float(net, "error", err_sum / self->n_channels);
      bptt_log_float(net, "correct", winners / self->n_channels);
      self->net->generation = net->generation;
    }
    send_message(self, err_sum / self->n_channels);

    self->incoming_start += chunk_size;
    self->incoming_start %= self->queue_size;
    len -= chunk_size;
  }
}



static GstFlowReturn
gst_classify_transform_ip (GstBaseTransform * base, GstBuffer *buf)
{
  GstClassify *self = GST_CLASSIFY(base);
  GstFlowReturn ret = GST_FLOW_OK;
  queue_audio_segment(buf, self->incoming_queue, self->queue_size,
      &self->incoming_start, &self->incoming_end);
  maybe_learn(self);
  GST_LOG("classify_transform returning OK");
  return ret;
}


static gboolean
plugin_init (GstPlugin * plugin)
{
  gboolean classify = gst_element_register (plugin, "classify", GST_RANK_NONE,\
      GST_TYPE_CLASSIFY);
  return classify;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    classify,
    "Classify audio streams",
    plugin_init, VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN);
