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

#include "pending_properties.h"

/* GstClassify signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

/* CLASSIFY_MODE will switch to TRAINING_MODE when valid targets are given.
   STICKY_CLASSIFY_MODE won't.
*/
enum
{
  STICKY_CLASSIFY_MODE = -1,
  CLASSIFY_MODE,
  TRAINING_MODE
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
  PROP_MOMENTUM_STYLE,
  PROP_MOMENTUM_SOFT_START,
  PROP_MFCCS,
  PROP_SAVE_NET,
  PROP_PGM_DUMP,
  PROP_LOG_FILE,
  PROP_LOG_CLASS_NUMBERS,
  PROP_MODE,
  PROP_WINDOW_SIZE,
  PROP_BASENAME,
  PROP_DROPOUT,
  PROP_ERROR_WEIGHT,
  PROP_BPTT_DEPTH,
  PROP_WEIGHT_SPARSITY,
  PROP_WEIGHT_FAN_IN_SUM,
  PROP_WEIGHT_FAN_IN_KURTOSIS,
  PROP_LAWN_MOWER,
  PROP_RNG_SEED,

  PROP_LAST
};

#define DEFAULT_PROP_TARGET ""
#define DEFAULT_PROP_PGM_DUMP ""
#define DEFAULT_PROP_LOG_FILE ""
#define DEFAULT_PROP_ERROR_WEIGHT ""
#define DEFAULT_BASENAME "classify"
#define DEFAULT_PROP_SAVE_NET NULL
#define DEFAULT_PROP_LOG_CLASS_NUMBERS 0
#define DEFAULT_PROP_LAWN_MOWER 0
#define DEFAULT_PROP_MODE 0
#define DEFAULT_PROP_MFCCS 0
#define DEFAULT_PROP_MOMENTUM 0.95f
#define DEFAULT_PROP_MOMENTUM_SOFT_START 0.0f
#define DEFAULT_PROP_MOMENTUM_STYLE 1
#define DEFAULT_PROP_CLASSES 2
#define DEFAULT_PROP_BPTT_DEPTH 30
#define DEFAULT_PROP_FORGET 0
#define DEFAULT_PROP_WEIGHT_SPARSITY 1
#define DEFAULT_WINDOW_SIZE 256
#define DEFAULT_HIDDEN_SIZE 199
#define DEFAULT_LEARN_RATE 0.0001
#define DEFAULT_PROP_DROPOUT 0.0f
#define MIN_PROP_CLASSES 1
#define MAX_PROP_CLASSES 1000000
#define MIN_PROP_BPTT_DEPTH 1
#define MAX_PROP_BPTT_DEPTH 1000
#define MIN_HIDDEN_SIZE 1
#define MAX_HIDDEN_SIZE 1000000
#define WINDOW_SIZE_MAX 8192
#define WINDOW_SIZE_MIN 32
#define LEARN_RATE_MIN 0.0
#define LEARN_RATE_MAX 1.0
#define MOMENTUM_MIN 0.0
#define MOMENTUM_MAX 1.0
#define MOMENTUM_STYLE_MIN 0
#define MOMENTUM_STYLE_MAX 2
#define WEIGHT_SPARSITY_MIN 0
#define WEIGHT_SPARSITY_MAX 10
#define DEFAULT_PROP_WEIGHT_FAN_IN_SUM 0
#define DEFAULT_PROP_WEIGHT_FAN_IN_KURTOSIS 0.3
#define PROP_WEIGHT_FAN_IN_SUM_MAX 99.0
#define PROP_WEIGHT_FAN_IN_SUM_MIN 0.0
#define PROP_WEIGHT_FAN_IN_KURTOSIS_MAX 1.5
#define PROP_WEIGHT_FAN_IN_KURTOSIS_MIN 0.0

#define DROPOUT_MIN 0.0
#define DROPOUT_MAX 1.0
#define MIN_PROP_MFCCS 0
#define MAX_PROP_MFCCS (CLASSIFY_N_FFT_BINS - 1)
#define MOMENTUM_SOFT_START_MAX 1e9
#define MOMENTUM_SOFT_START_MIN 0

#define DEFAULT_RNG_SEED 11

/* static_functions */
static void gst_classify_class_init(GstClassifyClass *g_class);
static void gst_classify_init(GstClassify *self);
static void gst_classify_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_classify_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static GstFlowReturn gst_classify_transform_ip(GstBaseTransform *base, GstBuffer *buf);
static gboolean gst_classify_setup(GstAudioFilter * filter, const GstAudioInfo * info);
static void maybe_parse_target_string(GstClassify *self);
static void maybe_start_logging(GstClassify *self);

#define gst_classify_parent_class parent_class
G_DEFINE_TYPE (GstClassify, gst_classify, GST_TYPE_AUDIO_FILTER)

#define CLASSIFY_CAPS_STRING "audio/x-raw, format = (string) " QUOTE(CLASSIFY_FORMAT) \
      ", rate = (int) " QUOTE(CLASSIFY_RATE) \
      ", channels = (int) [ " QUOTE(CLASSIFY_MIN_CHANNELS) " , " QUOTE(CLASSIFY_MAX_CHANNELS) " ] " \
      ", layout = (string) interleaved, channel-mask = (bitmask)0x0"


static inline void
init_channel(ClassifyChannel *c, RecurNN *net, int window_size, int id, float learn_rate)
{
  u32 flags = net->flags & ~RNN_NET_FLAG_OWN_WEIGHTS;
  c->net = rnn_clone(net, flags, RECUR_RNG_SUBSEED, NULL);
  c->net->bptt->learn_rate = learn_rate;
  c->pcm_next = zalloc_aligned_or_die(window_size * sizeof(float));
  c->pcm_now = zalloc_aligned_or_die(window_size * sizeof(float));
  c->features = zalloc_aligned_or_die(net->input_size * sizeof(float));
  if (PGM_DUMP_FEATURES && id == 0){
#if 0
    c->mfcc_image = temporal_ppm_alloc(net->i_size * net->bptt->depth, 300, "mfcc", id,
        PGM_DUMP_COLOUR, &c->net->bptt->history);
#else
    c->mfcc_image = temporal_ppm_alloc(net->input_size, 300, "mfcc", id,
        PGM_DUMP_COLOUR, &c->features);
#endif
  }
  else {
    c->mfcc_image = NULL;
  }
}

static inline void
finalise_channel(ClassifyChannel *c)
{
  if (c->net)
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
  if (self->mfcc_factory){
    recur_audio_binner_delete(self->mfcc_factory);
  }
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
  if (self->net){
    rnn_save_net(self->net, self->net_filename, 1);
    rnn_delete_net(self->net);
  }
  for (int i = 0; i < PROP_LAST; i++){
    GValue *v = PENDING_PROP(self, i);
    if (G_IS_VALUE(v)){
      g_value_unset(v);
    }
  }
  free(self->pending_properties);
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
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_BASENAME,
      g_param_spec_string("basename", "basename",
          "Base net file names on this root",
          DEFAULT_BASENAME,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_CLASSES,
      g_param_spec_int("classes", "classes",
          "Use this many classes",
          MIN_PROP_CLASSES, MAX_PROP_CLASSES,
          DEFAULT_PROP_CLASSES,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_BPTT_DEPTH,
      g_param_spec_int("bptt-depth", "bptt-depth",
          "Backprop through time to this depth",
          MIN_PROP_BPTT_DEPTH, MAX_PROP_BPTT_DEPTH,
          DEFAULT_PROP_BPTT_DEPTH,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MFCCS,
      g_param_spec_int("mfccs", "mfccs",
          "Use this many MFCCs, or zero for fft bins",
          MIN_PROP_MFCCS, MAX_PROP_MFCCS,
          DEFAULT_PROP_MFCCS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_WEIGHT_SPARSITY,
      g_param_spec_int("weight-sparsity", "weight-sparsity",
          "higher numbers for more initial weights near zero",
          WEIGHT_SPARSITY_MIN, WEIGHT_SPARSITY_MAX,
          DEFAULT_PROP_WEIGHT_SPARSITY,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_FORGET,
      g_param_spec_boolean("forget", "forget",
          "Forget the current hidden layer (all channels)",
          DEFAULT_PROP_FORGET,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LOG_CLASS_NUMBERS,
      g_param_spec_boolean("log-class-numbers", "log-class-numbers",
          "Log counts of each class in training",
          DEFAULT_PROP_LOG_CLASS_NUMBERS,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE,
      g_param_spec_int("mode", "mode",
          "toggle training: 0 - no training, 1 training, -1 sticky no training",
          STICKY_CLASSIFY_MODE, TRAINING_MODE,
          DEFAULT_PROP_MODE,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LEARN_RATE,
      g_param_spec_float("learn-rate", "learn-rate",
          "Learning rate for the RNN",
          LEARN_RATE_MIN, LEARN_RATE_MAX,
          DEFAULT_LEARN_RATE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_DROPOUT,
      g_param_spec_float("dropout", "dropout",
          "dropout this portion of neurons in training",
          DROPOUT_MIN, DROPOUT_MAX,
          DEFAULT_PROP_DROPOUT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_WEIGHT_FAN_IN_SUM,
      g_param_spec_float("weight-fan-in-sum", "weight-fan-in-sum",
          "If non-zero, initialise weights fan in to this sum (try 2)",
          PROP_WEIGHT_FAN_IN_SUM_MIN,
          PROP_WEIGHT_FAN_IN_SUM_MAX,
          DEFAULT_PROP_WEIGHT_FAN_IN_SUM,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_WEIGHT_FAN_IN_KURTOSIS,
      g_param_spec_float("weight-fan-in-kurtosis", "weight-fan-in-kurtosis",
          "degree of concentration of fan-in weights",
          PROP_WEIGHT_FAN_IN_KURTOSIS_MIN,
          PROP_WEIGHT_FAN_IN_KURTOSIS_MAX,
          DEFAULT_PROP_WEIGHT_FAN_IN_KURTOSIS,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

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
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MOMENTUM_STYLE,
      g_param_spec_int("momentum-style", "momentum-style",
          "0: hypersimplified Nesterov, 1: Nesterov, 2: classical momentum",
          MOMENTUM_STYLE_MIN, MOMENTUM_STYLE_MAX,
          DEFAULT_PROP_MOMENTUM_STYLE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_HIDDEN_SIZE,
      g_param_spec_int("hidden-size", "hidden-size",
          "Size of the RNN hidden layer",
          MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE,
          DEFAULT_HIDDEN_SIZE,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_WINDOW_SIZE,
      g_param_spec_int("window-size", "window-size",
          "Size of the input window (samples)",
          WINDOW_SIZE_MIN, WINDOW_SIZE_MAX,
          DEFAULT_WINDOW_SIZE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ERROR_WEIGHT,
      g_param_spec_string("error-weight", "error-weight",
          "Weight output errors (space or colon separated floats)",
          DEFAULT_PROP_ERROR_WEIGHT,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LAWN_MOWER,
      g_param_spec_boolean("lawn-mower", "lawn-mower",
          "Don't let any weight grow bigger than " QUOTE(RNN_LAWN_MOWER_THRESHOLD),
          DEFAULT_PROP_LAWN_MOWER,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_RNG_SEED,
      g_param_spec_uint64("rng-seed", "rng-seed",
          "RNG seed (only settable at start)",
          0, G_MAXUINT64,
          DEFAULT_RNG_SEED,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  trans_class->transform_ip = GST_DEBUG_FUNCPTR (gst_classify_transform_ip);
  af_class->setup = GST_DEBUG_FUNCPTR (gst_classify_setup);
  GST_INFO("gst audio class init\n");
}

static void
gst_classify_init (GstClassify * self)
{
  self->channels = NULL;
  self->n_channels = 0;
  self->mfcc_factory = NULL;
  self->incoming_queue = NULL;
  self->net_filename = NULL;
  self->pending_properties = calloc(PROP_LAST - 1, sizeof(GValue));
  self->class_events = NULL;
  self->class_events_index = 0;
  self->n_class_events = 0;
  self->learn_rate = DEFAULT_LEARN_RATE;
  self->window_size = DEFAULT_WINDOW_SIZE;
  self->momentum_soft_start = DEFAULT_PROP_MOMENTUM_SOFT_START;
  self->dropout = DEFAULT_PROP_DROPOUT;
  self->basename = strdup(DEFAULT_BASENAME);
  self->error_weight = NULL;
  GST_INFO("gst classify init\n");
}


static void
reset_net_filename(GstClassify *self, int hidden_size){
  char s[200];
  int n_features = self->mfccs ? self->mfccs : CLASSIFY_N_FFT_BINS;
  snprintf(s, sizeof(s), "%s-i%d-h%d-o%d-b%d-%dHz-w%d.net",
      self->basename, n_features, hidden_size, self->n_classes,
      CLASSIFY_BIAS, CLASSIFY_RATE, self->window_size);
  if (self->net_filename){
    free(self->net_filename);
  }
  self->net_filename = strdup(s);
}


static RecurNN *
load_or_create_net(GstClassify *self){
  int hidden_size = get_gvalue_int(PENDING_PROP(self, PROP_HIDDEN_SIZE),
      DEFAULT_HIDDEN_SIZE);
  if (self->net_filename == NULL)
    reset_net_filename(self, hidden_size);
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
    u32 flags = CLASSIFY_RNN_FLAGS;
    if (get_gvalue_boolean(PENDING_PROP(self, PROP_LAWN_MOWER),
            DEFAULT_PROP_LAWN_MOWER)){
      flags |= RNN_COND_USE_LAWN_MOWER;
    }
    else {
      flags &= ~RNN_COND_USE_LAWN_MOWER;
    }
    int weight_sparsity = get_gvalue_int(PENDING_PROP(self, PROP_WEIGHT_SPARSITY),
        DEFAULT_PROP_WEIGHT_SPARSITY);
    int bptt_depth = get_gvalue_int(PENDING_PROP(self, PROP_BPTT_DEPTH),
        DEFAULT_PROP_BPTT_DEPTH);
    float momentum = get_gvalue_float(PENDING_PROP(self, PROP_MOMENTUM),
        DEFAULT_PROP_MOMENTUM);

    float fan_in_sum = get_gvalue_float(PENDING_PROP(self,
            PROP_WEIGHT_FAN_IN_SUM),
        DEFAULT_PROP_WEIGHT_FAN_IN_SUM);
    float fan_in_kurtosis = get_gvalue_float(PENDING_PROP(self,
            PROP_WEIGHT_FAN_IN_KURTOSIS),
        DEFAULT_PROP_WEIGHT_FAN_IN_KURTOSIS);

    u64 rng_seed = get_gvalue_u64(PENDING_PROP(self, PROP_RNG_SEED), DEFAULT_RNG_SEED);
    STDERR_DEBUG("rng seed %lu", rng_seed);

    net = rnn_new(n_features, hidden_size,
        self->n_classes, flags, rng_seed,
        NULL, bptt_depth, self->learn_rate, momentum,
        CLASSIFY_BATCH_SIZE);
    if (fan_in_sum){
      rnn_randomise_weights_fan_in(net, fan_in_sum, fan_in_kurtosis, 0.1f, 0);
    }
    else {
      rnn_randomise_weights(net, RNN_INITIAL_WEIGHT_VARIANCE_FACTOR / net->h_size,
          weight_sparsity, 0.5);
    }
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
  if (self->mfcc_factory == NULL){
    self->mfcc_factory = recur_audio_binner_new(self->window_size,
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

  if (self->n_channels != info->channels){
    DEBUG("given %d channels, previous %d", info->channels, self->n_channels);
    if (self->incoming_queue){
      free(self->incoming_queue);
    }
    self->n_channels = info->channels;
    self->queue_size = info->channels * self->window_size * CLASSIFY_QUEUE_FACTOR;
    self->incoming_queue = malloc_aligned_or_die(self->queue_size * sizeof(s16));
    if (self->channels){
      free(self->channels);
    }
    self->channels = malloc_aligned_or_die(self->n_channels * sizeof(ClassifyChannel));
    if (self->subnets){
      free(self->subnets);
    }
    self->subnets = malloc_aligned_or_die(self->n_channels * sizeof(RecurNN *));
    for (int i = 0; i < self->n_channels; i++){
      init_channel(&self->channels[i], self->net, self->window_size,
          i, self->learn_rate);
      self->subnets[i] = self->channels[i].net;
    }
  }
  maybe_start_logging(self);
  maybe_parse_target_string(self);

  GstStructure *s = gst_structure_new_empty("classify-setup");
  GstMessage *msg = gst_message_new_element(GST_OBJECT(self), s);
  gst_element_post_message(GST_ELEMENT(self), msg);

  return TRUE;
}

static int
cmp_class_event(const void *a, const void *b){
  const int at = ((const ClassifyClassEvent *)a)->window_no;
  const int bt = ((const ClassifyClassEvent *)b)->window_no;
  return at - bt;
}

/*"complex" target specification.
  class   := 'c'<int>
  time    := 't'<float>
  event   := <class><time>
  channel := <event>*
  space   := ' '
  target_string := <channel><space><channel>;...

  There can be zero or more events per channel (though zero is rather useless
  for training).

  The number of channels should match self->n_channels.
 */
static inline int
parse_complex_target_string(GstClassify *self, const char *str){
  char *e;
  const char *s;
  int i;
  int channel = 0;
  ClassifyClassEvent *ev;
  /* the number of 'c's in the string is the number of class events */
  int n = 0;
  int nc = 0;
  for (s = str; *s; s++){
    n += *s == 'c';
    nc += *s == ' ';
  }
  if (self->n_class_events < n){
    GST_LOG("found %d targets, %d channels, prev %d events (%p)", n, nc,
        self->n_class_events, self->class_events);
    self->class_events = realloc_or_die(self->class_events,
        (n + 2) * sizeof(ClassifyClassEvent));
  }
  self->n_class_events = n;
  self->class_events_index = 0;
  GST_LOG("events %p, n %d", self->class_events, n);
  s = str;
  float time_to_window_no = CLASSIFY_RATE * 2.0f / self->window_size + 0.5;
  for (i = 0; i < n; i++){
#define ERROR_IF(x) if (x) goto parse_error
    ev = &self->class_events[i];
    ev->channel = channel;
    //GST_LOG("i %d s '%s' ev %p", i, s, ev);
    ERROR_IF(*s != 'c');
    s++;
    ev->class = strtol(s, &e, 10);
    ERROR_IF(s == e || *e != 't');    /*no digits at all, no comma, or out of range */
    ERROR_IF(ev->class < 0 || ev->class >= self->n_classes);
    s = e + 1;
    float time = strtod(s, &e);
    ev->window_no = time * time_to_window_no;
    ERROR_IF(s == e || ev->window_no < 0);

    s = e;
    if (*s == ' '){
      channel++;
      s++;
    }
    GST_LOG("event: channel %d target %d window %d starting %.2f (request %.2f)",
        ev->channel, ev->class, ev->window_no,
        (double)ev->window_no * self->window_size / (2.0f * CLASSIFY_RATE), time);
#undef ERROR_IF
  }
  qsort(self->class_events, n, sizeof(ClassifyClassEvent), cmp_class_event);
  self->class_events_index = 0;
  return 0;
 parse_error:
  GST_WARNING("Can't parse '%s' into %d events for %d channels: "
      "stopping after %d events (%ld chars)",
      s, n, self->n_channels, i, s - str);
  return -1;
}

static inline int
parse_simple_target_string(GstClassify *self, const char *s){
  char *e;
  long x;
  int i;
  for (i = 0; i < self->n_channels; i++){
    x = strtol(s, &e, 10);
    GST_LOG("channel %d got %ld s is %s, %p  e is %p", i, x, s, s, e);
    if (s == e){ /*no digits at all */
      goto parse_error;
    }
    self->channels[i].current_target = x;
    s = e + 1;
    if (*e == '\0'  && i != self->n_channels - 1){ /*ran out of numbers */
      goto parse_error;
    }
  }
  GST_LOG("target %d", self->channels[0].current_target);
  return 0;
 parse_error:
  GST_WARNING("Can't parse '%s' into %d channels: stopping after %d",
      s, self->n_channels, i);
  return 1;
}

static inline void
reset_channel_targets(GstClassify *self){
  int i;
  /*out of bounds [0, n_channels - 1) signals no target */
  for (i = 0; i < self->n_channels; i++){
    self->channels[i].current_target = -1;
  }
  if (self->mode == TRAINING_MODE){
    self->mode = CLASSIFY_MODE;
  }
}

static void
maybe_parse_target_string(GstClassify *self){
  char *target_string = steal_gvalue_string(PENDING_PROP(self, PROP_TARGET));
  char *s = target_string;
  if (s == NULL || self->channels == NULL){
    GST_DEBUG("not parsing target string (%p); channels is %p",
        s, self->channels);
    return;
  }
  GST_DEBUG("parsing target '%s'", s);
  if (*s == 0){
    reset_channel_targets(self);
  }
  else {
    int result;
    if (*s == 'c'){
      result = parse_complex_target_string(self, s);
    }
    else {
      result = parse_simple_target_string(self, s);
    }
    if (self->mode != STICKY_CLASSIFY_MODE){
      self->mode = (result == 0) ? TRAINING_MODE : CLASSIFY_MODE;
    }
  }
  free(target_string);
  self->window_no = 0;
}

static void
maybe_parse_error_weight_string(GstClassify *self){
  char *orig, *e, *s;
  e = orig = s = steal_gvalue_string(PENDING_PROP(self, PROP_ERROR_WEIGHT));
  int i;
  if (s == NULL || self->channels == NULL){
    GST_DEBUG("not parsing error_weight string:"
        "either it (%p) or channels (%p) is NULL",
        s, self->channels);
     return;
   }
  GST_DEBUG("parsing error weights '%s'", s);
  if (*s == 0){
    if (self->error_weight){
      free(self->error_weight);
      self->error_weight = 0;
    }
  }
  else {
    int len = self->net->output_size;
    if (self->error_weight == NULL){
      self->error_weight = malloc_aligned_or_die(len);
    }
    for (i = 0; i < len && *e && *s; i++, s = e + 1){
      self->error_weight[i] = strtof(s, &e);
    }
    if (i < len){
      GST_WARNING("error weight string property is short"
          "found %d numbers, wanted %d in '%s'", i, len, orig);
    }
  }
  free(orig);
}

static void
maybe_start_logging(GstClassify *self){
  const char *s = get_gvalue_string(PENDING_PROP(self, PROP_LOG_FILE));
  GST_DEBUG("pending log '%s'; subnets is %p", s, self->subnets);
  if (s && self->subnets){
    if (s[0] == 0){
      rnn_set_log_file(self->subnets[0], NULL, 0);
    }
    else {
      rnn_set_log_file(self->subnets[0], s, 1);
    }
    g_value_unset(PENDING_PROP(self, PROP_LOG_FILE));
  }
}

/*maybe_set_mode sets self->mode to the requested boolean value UNLESS true is
  requested while the targets are invalid. It returns the actually set value.
*/

static int
maybe_set_mode(GstClassify *self, int t){
  if (t == TRAINING_MODE){
    for (int i = 0; i < self->n_channels; i++){
      ClassifyChannel *c = &self->channels[i];
      if (c->current_target < 0 ||
          c->current_target >= self->n_classes){
        GST_DEBUG("asked for training mode, but target %d is bad (%d)",
            i, c->current_target);
        t = CLASSIFY_MODE;
        break;
      }
    }
  }
  else if (t != CLASSIFY_MODE && t != STICKY_CLASSIFY_MODE){
    GST_WARNING("asked for invalid mode %d, using CLASSIFY_MODE instead", t);
    t = CLASSIFY_MODE;
  }
  GST_DEBUG("Setting mode to %d", t);
  self->mode = t;
  return t;
}

static void
gst_classify_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstClassify *self = GST_CLASSIFY (object);
  GST_DEBUG("gst_classify_set_property with prop_id %d\n", prop_id);
  if (value){
    const char *strvalue;
    switch (prop_id) {
    case PROP_TARGET:
      set_gvalue(PENDING_PROP(self, prop_id), value);
      maybe_parse_target_string(self);
      break;

    case PROP_ERROR_WEIGHT:
      set_gvalue(PENDING_PROP(self, prop_id), value);
      maybe_parse_error_weight_string(self);
      break;

    case PROP_PGM_DUMP:
      if (self->net){
        strvalue = g_value_get_string(value);
        rnn_multi_pgm_dump(self->net, strvalue);
      }
      break;

    case PROP_BASENAME:
      free(self->basename);
      self->basename = g_value_dup_string(value);
      break;

    case PROP_SAVE_NET:
      strvalue = g_value_get_string(value);
      if (self->net){
        if (strvalue && strvalue[0] != 0){
          rnn_save_net(self->net, strvalue, 1);
        }
        else {
          rnn_save_net(self->net, self->net_filename, 1);
        }
      }
      break;

    case PROP_LOG_FILE:
      /*defer setting the actual log file, in case the nets aren't ready yet*/
      set_gvalue(PENDING_PROP(self, prop_id), value);
      maybe_start_logging(self);
      break;

    case PROP_FORGET:
      if (self->net && self->channels){
        gboolean bptt_too = g_value_get_boolean(value);
        rnn_forget_history(self->net, bptt_too);
        for (int i = 0; i < self->n_channels; i++){
          rnn_forget_history(self->channels[i].net, bptt_too);
        }
      }
      break;

    case PROP_LOG_CLASS_NUMBERS:
      self->log_class_numbers = g_value_get_boolean(value);
      break;

    case PROP_MODE:
      maybe_set_mode(self, g_value_get_int(value));
      break;

    case PROP_DROPOUT:
      self->dropout = g_value_get_float(value);
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

    case PROP_MOMENTUM_STYLE:
      self->momentum_style = g_value_get_int(value);
      break;

    /*properties that only need to be stored until net creation, and can't
      be changed afterwards go here.
    */
    case PROP_MOMENTUM:
    case PROP_HIDDEN_SIZE:
    case PROP_BPTT_DEPTH:
    case PROP_LAWN_MOWER:
    case PROP_WEIGHT_SPARSITY:
    case PROP_WEIGHT_FAN_IN_SUM:
    case PROP_WEIGHT_FAN_IN_KURTOSIS:
    case PROP_RNG_SEED:
      if (self->net == NULL){
        set_gvalue(PENDING_PROP(self, prop_id), value);
      }
      else {
        GST_WARNING("it is TOO LATE to set %s.", pspec->name);
      }
      break;

      /*these next ones have no effect if set late (after net creation)
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

    case PROP_WINDOW_SIZE:
      SET_INT_IF_NOT_TOO_LATE(window_size, "audio window size");
      break;

    case PROP_MFCCS:
      SET_INT_IF_NOT_TOO_LATE(mfccs, "number of MFCCs");
      break;

    case PROP_CLASSES:
      SET_INT_IF_NOT_TOO_LATE(n_classes, "number of classes");
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
  case PROP_DROPOUT:
    g_value_set_float(value, self->dropout);
    break;
  case PROP_MOMENTUM_SOFT_START:
    g_value_set_float(value, self->momentum_soft_start);
    break;
  case PROP_MOMENTUM_STYLE:
    g_value_set_int(value, self->momentum_style);
    break;
  case PROP_WINDOW_SIZE:
    g_value_set_int(value, self->window_size);
    break;
  case PROP_LAWN_MOWER:
    {
      gboolean x;
      if (self->net){
        x = !! (self->net->flags & RNN_COND_USE_LAWN_MOWER);
      }
      else {
        x = get_gvalue_boolean(PENDING_PROP(self, PROP_LAWN_MOWER),
            DEFAULT_PROP_LAWN_MOWER);
      }
      g_value_set_boolean(value, x);
    }
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


static inline void
send_message(GstClassify *self, float mean_err, GstClockTime pts)
{
  GstMessage *msg;
  char *name = "classify";
  GstStructure *s;
  s = gst_structure_new (name,
      "error", G_TYPE_FLOAT, mean_err,
      NULL);
  for (int i = 0; i < self->n_channels; i++){
    char key[50];
    ClassifyChannel *c = &self->channels[i];
    RecurNN *net = c->net;
    for (int j = 0; j < net->output_size; j++){
      snprintf(key, sizeof(key), "channel %d, output %d", i, j);
      gst_structure_set(s, key, G_TYPE_FLOAT, net->bptt->o_error[j], NULL);
    }
    if (c->current_target >= 0 &&
        c->current_target < self->n_classes){
      snprintf(key, sizeof(key), "channel %d correct", i);
      gst_structure_set(s, key, G_TYPE_INT, c->current_winner == c->current_target, NULL);
      snprintf(key, sizeof(key), "channel %d target", i);
      gst_structure_set(s, key, G_TYPE_INT, c->current_target, NULL);
    }
    snprintf(key, sizeof(key), "channel %d winner", i);
    gst_structure_set(s, key, G_TYPE_INT, c->current_winner, NULL);
  }
  msg = gst_message_new_element(GST_OBJECT(self), s);
  msg->timestamp = pts;
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
  /*XXX could do at least one less copy */
  for (int i = 0; i < n_features; i++){
    features[i] = answer[i];
  }
}

static inline ClassifyChannel *
prepare_channel_features(GstClassify *self, s16 *buffer_i, int j){
  /*load first half of pcm_next, second part of pcm_now.*/
  /*second part of pcm_next retains previous data */
  int i, k;
  int half_window = self->window_size / 2;
  ClassifyChannel *c = &self->channels[j];
  for(i = 0, k = j; i < half_window; i++, k += self->n_channels){
    c->pcm_next[i] = buffer_i[k];
    c->pcm_now[half_window + i] = buffer_i[k];
  }
  /*get the features -- after which pcm_now is finished with. */
  pcm_to_features(self->mfcc_factory, c->features, c->pcm_now, self->mfccs);
  if (c->mfcc_image){
    temporal_ppm_row_from_source(c->mfcc_image);
  }

  float *tmp;
  tmp = c->pcm_next;
  c->pcm_next = c->pcm_now;
  c->pcm_now = tmp;
  return c;
}

static inline float
train_channel(ClassifyChannel *c, float dropout, float *error_weights){
  RecurNN *net = c->net;
  float *answer;
  if (dropout){
    answer = rnn_opinion_with_dropout(net, c->features, dropout);
  }
  else {
    answer = rnn_opinion(net, c->features);
  }
  c->current_winner = softmax_best_guess(net->bptt->o_error, answer,
      net->output_size);
  net->bptt->o_error[c->current_target] += 1.0f;
  if (error_weights){
    GST_DEBUG("error weights %f %f", error_weights[0], error_weights[1]);
    for (int i = 0; i < net->output_size; i ++){
      net->bptt->o_error[i] *= error_weights[i];
    }
  }
  rnn_bptt_calc_deltas(net, NULL, NULL);
  rnn_bptt_advance(net);
  return net->bptt->o_error[c->current_target];
}

static inline s16 *
prepare_next_chunk(GstClassify *self){
  /*change the target classes of channels where necessary*/
  int half_window = self->window_size / 2;
  int chunk_size = half_window * self->n_channels;
  int len = self->incoming_end - self->incoming_start;
  if (len < 0){
    len += self->queue_size;
  }
  GST_LOG("start %d end %d len %d chunk %d queue_size %d", self->incoming_start,
      self->incoming_end, len, chunk_size, self->queue_size);
  if (len < chunk_size){
    GST_LOG("returning NULL");
    return NULL;
  }
  self->incoming_start += chunk_size;
  if (self->incoming_start >= self->queue_size){
    self->incoming_start -= self->queue_size;
  }

  while(self->class_events_index < self->n_class_events){
    ClassifyClassEvent *ev = &self->class_events[self->class_events_index];
    if (ev->window_no > self->window_no){
      break;
    }
    self->channels[ev->channel].current_target = ev->class;
    GST_LOG("event %d/%d: channel %d target -> %d at window  %d (%d)",
        self->class_events_index, self->n_class_events,
        ev->channel, ev->class, self->window_no, ev->window_no);
    self->class_events_index++;
  }

  self->window_no++;
  GST_LOG("returning %p", self->incoming_queue + self->incoming_start);

  return self->incoming_queue + self->incoming_start;
}

static inline void
maybe_learn(GstClassify *self){
  int i, j;
  s16 *buffer;
  while ((buffer = prepare_next_chunk(self))){
    float err_sum = 0.0f;
    float winners = 0.0f;
    int class_counts[self->n_classes];
    for (i = 0; i < self->n_classes; i++){
      class_counts[i] = 0;
    }
    for (j = 0; j < self->n_channels; j++){
      ClassifyChannel *c = prepare_channel_features(self, buffer, j);
      err_sum += train_channel(c, self->dropout, self->error_weight);
      winners += c->current_winner == c->current_target;
      class_counts[c->current_target]++;
    }

    RecurNN *net = self->subnets[0];
    /*XXX periodic_pgm_dump and image string should be gst properties */
    if (PERIODIC_PGM_DUMP && net->generation % PERIODIC_PGM_DUMP == 0){
      rnn_multi_pgm_dump(net, "how ihw");
    }
    rnn_consolidate_many_nets(self->subnets, self->n_channels, self->momentum_style,
        self->momentum_soft_start);
    rnn_condition_net(self->net);
    possibly_save_net(self->net, self->net_filename);
    rnn_log_net(net);
    if (self->log_class_numbers){
      for (i = 0; i < self->n_classes; i++){
        char s[20];
        snprintf(s, sizeof(s), "class-%d", i);
        rnn_log_int(net, s, class_counts[i]);
      }
    }
    rnn_log_float(net, "error", err_sum / self->n_channels);
    rnn_log_float(net, "correct", winners / self->n_channels);
    self->net->generation = net->generation;
  }
}

static inline void
emit_opinions(GstClassify *self, GstClockTime pts){
  int j;
  s16 *buffer;
  for (buffer = prepare_next_chunk(self); buffer;
       buffer = prepare_next_chunk(self)){
    float err_sum = 0.0f;
    for (j = 0; j < self->n_channels; j++){
      ClassifyChannel *c = prepare_channel_features(self, buffer, j);
      RecurNN *net = c->net;
      float *answer = rnn_opinion(net, c->features);
      c->current_winner = softmax_best_guess(net->bptt->o_error, answer,
          net->output_size);
      int valid_target = c->current_target >= 0 && c->current_target < self->n_classes;
      if (valid_target){
        err_sum += net->bptt->o_error[c->current_target];
      }
    }
    send_message(self, err_sum / self->n_channels, pts);
  }
}


static GstFlowReturn
gst_classify_transform_ip (GstBaseTransform * base, GstBuffer *buf)
{
  GstClassify *self = GST_CLASSIFY(base);
  GstFlowReturn ret = GST_FLOW_OK;
  queue_audio_segment(buf, self->incoming_queue, self->queue_size,
      &self->incoming_start, &self->incoming_end);
  if (self->mode == TRAINING_MODE){
    maybe_learn(self);
  }
  else {
    emit_opinions(self, GST_BUFFER_PTS(buf));
  }
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
    plugin_init, VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
