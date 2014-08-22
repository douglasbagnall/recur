/* Copyright 2013 Douglas Bagnall <douglas@halo.gen.nz> LGPL */

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

enum
{
  PROP_0,
  PROP_TARGET,
  PROP_CLASSES,
  PROP_FORGET,
  PROP_LEARN_RATE,
  PROP_TOP_LEARN_RATE_SCALE,
  PROP_BOTTOM_LEARN_RATE_SCALE,
  PROP_HIDDEN_SIZE,
  PROP_MIN_FREQUENCY,
  PROP_MAX_FREQUENCY,
  PROP_KNEE_FREQUENCY,
  PROP_FOCUS_FREQUENCY,
  PROP_MOMENTUM,
  PROP_LEARNING_STYLE,
  PROP_MOMENTUM_SOFT_START,
  PROP_MFCCS,
  PROP_SAVE_NET,
  PROP_PGM_DUMP,
  PROP_LOG_FILE,
  PROP_TRAINING,
  PROP_WINDOW_SIZE,
  PROP_BASENAME,
  PROP_ERROR_WEIGHT,
  PROP_BPTT_DEPTH,
  PROP_WEIGHT_INIT_METHOD,
  PROP_WEIGHT_FAN_IN_SUM,
  PROP_WEIGHT_FAN_IN_KURTOSIS,
  PROP_LAWN_MOWER,
  PROP_RNG_SEED,
  PROP_BOTTOM_LAYER,
  PROP_RANDOM_ALIGNMENT,
  PROP_NET_FILENAME,
  PROP_DELTA_FEATURES,
  PROP_INTENSITY_FEATURE,
  PROP_FORCE_LOAD,
  PROP_LAG,
  PROP_IGNORE_START,
  PROP_GENERATION,
  PROP_WEIGHT_NOISE,
  PROP_WEIGHT_INIT_SCALE,
  PROP_CONFIRMATION_LAG,
  PROP_LOAD_NET_NOW,
  PROP_WINDOWS_PER_SECOND,
  PROP_PRESYNAPTIC_NOISE,
  PROP_ADAGRAD_BALLAST,

  PROP_LAST
};

#define DEFAULT_PROP_TARGET ""
#define DEFAULT_PROP_PGM_DUMP ""
#define DEFAULT_PROP_LOG_FILE ""
#define DEFAULT_PROP_ERROR_WEIGHT ""
#define DEFAULT_BASENAME "classify"
#define DEFAULT_PROP_SAVE_NET NULL
#define DEFAULT_PROP_LAWN_MOWER 0
#define DEFAULT_PROP_TRAINING 0
#define DEFAULT_PROP_MFCCS 0
#define DEFAULT_PROP_DELTA_FEATURES 0
#define DEFAULT_PROP_INTENSITY_FEATURE FALSE
#define DEFAULT_PROP_MOMENTUM 0.95f
#define DEFAULT_PROP_MOMENTUM_SOFT_START 0.0f
#define DEFAULT_PROP_LEARNING_STYLE 1
#define DEFAULT_MIN_FREQUENCY 100
#define DEFAULT_KNEE_FREQUENCY 700
#define DEFAULT_FOCUS_FREQUENCY 0
#define DEFAULT_MAX_FREQUENCY (CLASSIFY_RATE * 0.499)
#define MINIMUM_AUDIO_FREQUENCY 0
#define MAXIMUM_AUDIO_FREQUENCY (CLASSIFY_RATE * 0.5)
#define DEFAULT_PROP_WEIGHT_NOISE 0.0f
#define DEFAULT_PROP_WEIGHT_INIT_SCALE 0.0f
#define DEFAULT_PROP_GENERATION 0
#define DEFAULT_PROP_WINDOWS_PER_SECOND 0
#define DEFAULT_PROP_PRESYNAPTIC_NOISE 0
#define DEFAULT_PROP_ADAGRAD_BALLAST 200

#define DEFAULT_PROP_CLASSES "01"
#define DEFAULT_PROP_BPTT_DEPTH 30
#define DEFAULT_PROP_FORGET 0
#define DEFAULT_PROP_FORCE_LOAD 0
#define DEFAULT_PROP_BOTTOM_LAYER 0
#define DEFAULT_PROP_RANDOM_ALIGNMENT 1
#define DEFAULT_WINDOW_SIZE 256
#define DEFAULT_HIDDEN_SIZE 199
#define DEFAULT_LEARN_RATE 0.0001
#define DEFAULT_TOP_LEARN_RATE_SCALE 1.0f
#define DEFAULT_BOTTOM_LEARN_RATE_SCALE 1.0f
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
#define LEARNING_STYLE_MIN 0
#define LEARNING_STYLE_MAX (RNN_LAST_MOMENTUM_METHOD - 1)
#define DEFAULT_PROP_WEIGHT_FAN_IN_SUM 0
#define DEFAULT_PROP_WEIGHT_FAN_IN_KURTOSIS 0.3
#define DEFAULT_PROP_LAG 0
#define DEFAULT_PROP_CONFIRMATION_LAG 0
#define PROP_LAG_MIN -1
#define PROP_LAG_MAX 1000
#define DEFAULT_PROP_IGNORE_START 0
#define PROP_IGNORE_START_MIN 0
#define PROP_IGNORE_START_MAX 1e4

#define PROP_WEIGHT_INIT_METHOD_MIN RNN_INIT_FLAT
#define PROP_WEIGHT_INIT_METHOD_MAX (RNN_INIT_LAST - 1)
#define PROP_WEIGHT_FAN_IN_SUM_MAX 99.0
#define PROP_WEIGHT_FAN_IN_SUM_MIN 0.0
#define PROP_WEIGHT_FAN_IN_KURTOSIS_MAX 1.5
#define PROP_WEIGHT_FAN_IN_KURTOSIS_MIN 0.0
#define PROP_BOTTOM_LAYER_MIN 0
#define PROP_BOTTOM_LAYER_MAX 1000000

#define LEARN_RATE_SCALE_MAX 1e9f
#define LEARN_RATE_SCALE_MIN 0

#define PROP_DELTA_FEATURES_MIN 0
#define PROP_DELTA_FEATURES_MAX 4
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
static gboolean start(GstBaseTransform *trans);
static void maybe_parse_target_string(GstClassify *self);
static void maybe_start_logging(GstClassify *self);
static void maybe_parse_error_weight_string(GstClassify *self);


#define gst_classify_parent_class parent_class
G_DEFINE_TYPE (GstClassify, gst_classify, GST_TYPE_AUDIO_FILTER)

#define CLASSIFY_CAPS_STRING "audio/x-raw, format = (string) " QUOTE(CLASSIFY_FORMAT) \
      ", rate = (int) " QUOTE(CLASSIFY_RATE) \
      ", channels = (int) [ " QUOTE(CLASSIFY_MIN_CHANNELS) " , " QUOTE(CLASSIFY_MAX_CHANNELS) " ] " \
      ", layout = (string) interleaved, channel-mask = (bitmask)0x0"


static inline void
init_channel(ClassifyChannel *c, RecurNN *net,
    int window_size, int id, int n_groups, uint delta_depth)
{
  c->net = net;
  int n_inputs;
  if (net->bottom_layer){
    n_inputs = net->bottom_layer->input_size;
  }
  else {
    n_inputs = net->input_size;
  }

  c->pcm_next = zalloc_aligned_or_die(window_size * sizeof(float));
  c->pcm_now = zalloc_aligned_or_die(window_size * sizeof(float));

  c->features = zalloc_aligned_or_die(n_inputs * sizeof(float));
  if (delta_depth > 0){
    c->prev_features = zalloc_aligned_or_die(n_inputs * sizeof(float));
  }
  else {
    c->prev_features = NULL;
  }
  c->group_target = zalloc_aligned_or_die(n_groups * 2 * sizeof(int));
  c->group_winner = c->group_target + n_groups;
  c->mfcc_image = NULL;

  if (PGM_DUMP_FEATURES && id == 0){
    c->mfcc_image = temporal_ppm_alloc(n_inputs, 300, "features", id,
        PGM_DUMP_COLOUR, &c->features);
  }
}

static inline void
finalise_channel(ClassifyChannel *c)
{
  free(c->pcm_next);
  free(c->pcm_now);
  if (c->features){
    free(c->features);
  }
  if (c->prev_features){
    free(c->prev_features);
  }
  if (c->mfcc_image){
    temporal_ppm_free(c->mfcc_image);
    c->mfcc_image = NULL;
  }
  free(c->group_target);
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
  }
  if (self->subnets){
    rnn_delete_training_set(self->subnets, self->n_channels, 1);
    self->subnets = NULL;
  }
  if (self->net){
    rnn_save_net(self->net, self->net_filename, 1);
    rnn_delete_net(self->net);
    self->net = NULL;
  }
  if (self->audio_queue){
    free(self->audio_queue);
    self->audio_queue = NULL;
  }
  if (self->error_image){
    temporal_ppm_free(self->error_image);
    self->error_image = NULL;
  }

  for (int i = 0; i < PROP_LAST; i++){
    GValue *v = PENDING_PROP(self, i);
    if (G_IS_VALUE(v)){
      g_value_unset(v);
    }
  }
  free(self->pending_properties);
}

static gboolean
stop(GstBaseTransform *trans){
  //STDERR_DEBUG("in stop()");
  return 1;
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
  trans_class->start = GST_DEBUG_FUNCPTR (start);
  trans_class->stop = GST_DEBUG_FUNCPTR (stop);

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
          "target values for each channel (complex syntax)",
          DEFAULT_PROP_TARGET,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_NET_FILENAME,
      g_param_spec_string("net-filename", "net-filename",
          "Load net from here (and save here)",
          NULL,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_PGM_DUMP,
      g_param_spec_string("pgm-dump", "pgm-dump",
          "Dump weight images (space separated \"ih* hh* ho* bi*\", *one of \"wdmt\")",
          DEFAULT_PROP_PGM_DUMP,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SAVE_NET,
      g_param_spec_string("save-net", "save-net",
          "Save the net here, now. (empty/null for auto-naming)",
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
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_CLASSES,
      g_param_spec_string("classes", "classes",
          "Identify classes (one letter per class, groups separated by commas)",
          DEFAULT_PROP_CLASSES,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_BPTT_DEPTH,
      g_param_spec_int("bptt-depth", "bptt-depth",
          "Backprop through time to this depth",
          MIN_PROP_BPTT_DEPTH, MAX_PROP_BPTT_DEPTH,
          DEFAULT_PROP_BPTT_DEPTH,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MFCCS,
      g_param_spec_int("mfccs", "mfccs",
          "Use this many MFCCs, or zero for fft bins",
          MIN_PROP_MFCCS, MAX_PROP_MFCCS,
          DEFAULT_PROP_MFCCS,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_DELTA_FEATURES,
      g_param_spec_int("delta-features", "delta-features",
          "Include this many levels of derivative features",
          PROP_DELTA_FEATURES_MIN, PROP_DELTA_FEATURES_MAX,
          DEFAULT_PROP_DELTA_FEATURES,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_INTENSITY_FEATURE,
      g_param_spec_boolean("intensity-feature", "intensity-feature",
          "Use the total signal intensity as one input",
          DEFAULT_PROP_INTENSITY_FEATURE,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_FORGET,
      g_param_spec_boolean("forget", "forget",
          "Forget the current hidden layer (all channels)",
          DEFAULT_PROP_FORGET,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_FORCE_LOAD,
      g_param_spec_boolean("force-load", "force-load",
          "Force the net to load even if metadata doesn't match",
          DEFAULT_PROP_FORCE_LOAD,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_RANDOM_ALIGNMENT,
      g_param_spec_boolean("random-alignment", "random-alignment",
          "randomly offset beginning of audio frames",
          DEFAULT_PROP_RANDOM_ALIGNMENT,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_BOTTOM_LAYER,
      g_param_spec_int("bottom-layer", "bottom-layer",
          "Use a bottom layer",
          PROP_BOTTOM_LAYER_MIN,
          PROP_BOTTOM_LAYER_MAX,
          DEFAULT_PROP_BOTTOM_LAYER,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_TRAINING,
      g_param_spec_boolean("training", "training",
          "set to true to train",
          DEFAULT_PROP_TRAINING,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LAG,
      g_param_spec_float("lag", "lag",
          "Add this many seconds onto all timings",
          PROP_LAG_MIN, PROP_LAG_MAX,
          DEFAULT_PROP_LAG,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_CONFIRMATION_LAG,
      g_param_spec_float("confirmation-lag", "confirmation-lag",
          "Confirmation classification after this many seconds",
          PROP_LAG_MIN, PROP_LAG_MAX,
          DEFAULT_PROP_CONFIRMATION_LAG,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_IGNORE_START,
      g_param_spec_float("ignore-start", "ignore-start",
          "Ignore this many seconds at the beginning of the file",
          PROP_IGNORE_START_MIN, PROP_IGNORE_START_MAX,
          DEFAULT_PROP_IGNORE_START,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MIN_FREQUENCY,
      g_param_spec_float("min-frequency", "min-frequency",
          "Lowest audio frequency to analyse",
          MINIMUM_AUDIO_FREQUENCY, MAXIMUM_AUDIO_FREQUENCY,
          DEFAULT_MIN_FREQUENCY,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_KNEE_FREQUENCY,
      g_param_spec_float("knee-frequency", "knee-frequency",
          "lower for more pronounced knee (~ less high end)",
          MINIMUM_AUDIO_FREQUENCY, MAXIMUM_AUDIO_FREQUENCY,
          DEFAULT_KNEE_FREQUENCY,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_FOCUS_FREQUENCY,
      g_param_spec_float("focus-frequency", "focus-frequency",
          "concentrate on frequencies around here",
          MINIMUM_AUDIO_FREQUENCY, MAXIMUM_AUDIO_FREQUENCY,
          DEFAULT_FOCUS_FREQUENCY,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MAX_FREQUENCY,
      g_param_spec_float("max-frequency", "max-frequency",
          "Highest audio frequency to analyse",
          MINIMUM_AUDIO_FREQUENCY, MAXIMUM_AUDIO_FREQUENCY,
          DEFAULT_MAX_FREQUENCY,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LEARN_RATE,
      g_param_spec_float("learn-rate", "learn-rate",
          "Base learning rate for the RNN",
          LEARN_RATE_MIN, LEARN_RATE_MAX,
          DEFAULT_LEARN_RATE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_TOP_LEARN_RATE_SCALE,
      g_param_spec_float("top-learn-rate-scale", "top-learn-rate-scale",
          "learn rate scale for top layer",
          LEARN_RATE_SCALE_MIN, LEARN_RATE_SCALE_MAX,
          DEFAULT_TOP_LEARN_RATE_SCALE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_BOTTOM_LEARN_RATE_SCALE,
      g_param_spec_float("bottom-learn-rate-scale", "bottom-learn-rate-scale",
          "learn rate scale for bottom layer (if any)",
          LEARN_RATE_SCALE_MIN, LEARN_RATE_SCALE_MAX,
          DEFAULT_BOTTOM_LEARN_RATE_SCALE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_WEIGHT_INIT_METHOD,
      g_param_spec_int("weight-init-method", "weight-init-method",
          "initialisation method. 1:flat, 2:fan in, 3: runs or loops",
          PROP_WEIGHT_INIT_METHOD_MIN,
          PROP_WEIGHT_INIT_METHOD_MAX,
          RNN_INIT_FLAT, /* not used; uses recur-nn default.*/
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

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
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LEARNING_STYLE,
      g_param_spec_int("learning-style", "learning-style",
          "0: hypersimplified Nesterov, 1: Nesterov, 2: classical momentum",
          LEARNING_STYLE_MIN, LEARNING_STYLE_MAX,
          DEFAULT_PROP_LEARNING_STYLE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_HIDDEN_SIZE,
      g_param_spec_int("hidden-size", "hidden-size",
          "Size of the RNN hidden layer",
          MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE,
          DEFAULT_HIDDEN_SIZE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_WINDOW_SIZE,
      g_param_spec_int("window-size", "window-size",
          "Size of the input window (samples)",
          WINDOW_SIZE_MIN, WINDOW_SIZE_MAX,
          DEFAULT_WINDOW_SIZE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ERROR_WEIGHT,
      g_param_spec_string("error-weight", "error-weight",
          "Weight output errors (space, comma, or colon separated floats)",
          DEFAULT_PROP_ERROR_WEIGHT,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LAWN_MOWER,
      g_param_spec_boolean("lawn-mower", "lawn-mower",
          "Don't let any weight grow bigger than " QUOTE(RNN_LAWN_MOWER_THRESHOLD),
          DEFAULT_PROP_LAWN_MOWER,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LOAD_NET_NOW,
      g_param_spec_boolean("load-net-now", "load-net-now",
          "Load or create the net based on properites so far",
          0,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_RNG_SEED,
      g_param_spec_uint64("rng-seed", "rng-seed",
          "RNG seed (only settable at start)",
          0, G_MAXUINT64,
          DEFAULT_RNG_SEED,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_WEIGHT_NOISE,
      g_param_spec_float("weight-noise", "weight-noise",
          "Std dev of noise added to weights before each training cycle",
          -G_MAXFLOAT, G_MAXFLOAT,
          DEFAULT_PROP_WEIGHT_NOISE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_WEIGHT_INIT_SCALE,
      g_param_spec_float("weight-init-scale", "weight-init-scale",
          "Scale recurrent weights to approximately this gain (0: no scaling)",
          0, G_MAXFLOAT,
          DEFAULT_PROP_WEIGHT_INIT_SCALE,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_PRESYNAPTIC_NOISE,
      g_param_spec_float("presynaptic-noise", "presynaptic-noise",
          "Add this much noise before nonlinear tranform",
          0, G_MAXFLOAT,
          DEFAULT_PROP_PRESYNAPTIC_NOISE,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ADAGRAD_BALLAST,
      g_param_spec_float("adagrad-ballast", "adagrad-ballast",
          "Start with this gradient sum when using ADAGRAD",
          0, G_MAXFLOAT,
          DEFAULT_PROP_ADAGRAD_BALLAST,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_GENERATION,
      g_param_spec_uint("generation", "generation",
          "Read the net's training generation",
          0, G_MAXUINT,
          DEFAULT_PROP_GENERATION,
          G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_WINDOWS_PER_SECOND,
      g_param_spec_double("windows-per-second", "windows-per-second",
          "Expect this many classifications per second",
          0, G_MAXDOUBLE,
          DEFAULT_PROP_WINDOWS_PER_SECOND,
          G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

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
  self->audio_queue = NULL;
  self->net_filename = NULL;
  self->training = DEFAULT_PROP_TRAINING;
  self->pending_properties = calloc(PROP_LAST, sizeof(GValue));
  self->class_events = NULL;
  self->class_events_index = 0;
  self->n_class_events = 0;
  self->momentum_soft_start = DEFAULT_PROP_MOMENTUM_SOFT_START;
  self->weight_noise = DEFAULT_PROP_WEIGHT_NOISE;
  self->error_weight = NULL;
  self->ignored_windows = 0;
  GST_INFO("gst classify init\n");
}

static inline int
get_n_features(GstClassify *self){
  int n;
  if (self->mfccs){
    n = self->mfccs;
  }
  else {
    n = CLASSIFY_N_FFT_BINS;
  }
  n += self->intensity_feature;
  n *= (1 + self->delta_features);
  return n;
}


static void
set_net_filename(GstClassify *self, int hidden_size, int bottom_layer,
    int top_layer_size, char *metadata){
  char s[200];
  u32 sig = rnn_hash32(metadata);
  int n_features = get_n_features(self);
  if (bottom_layer > 0){
    snprintf(s, sizeof(s), "%s-%0" PRIx32 "-i%d-b%d-h%d-o%d-%dHz-w%d.net",
        self->basename, sig, n_features, bottom_layer, hidden_size, top_layer_size,
        CLASSIFY_RATE, self->window_size);
  }
  else {
    snprintf(s, sizeof(s), "%s-%0" PRIx32 "-i%d-h%d-o%d-%dHz-w%d.net",
        self->basename, sig, n_features, hidden_size, top_layer_size,
        CLASSIFY_RATE, self->window_size);
  }
  self->net_filename = strdup(s);
}



static inline int
count_class_groups(const char *s){
  int n_groups = 1;
  for (; *s; s++){
    n_groups += (*s == ',');
  }
  return n_groups;
}

static inline int
count_class_group_members(const char *s){
  int n_classes = 0;
  for (; *s; s++){
    n_classes += (*s != ',');
  }
  return n_classes;
}

static int parse_classes_string(GstClassify *self, const char *orig)
{
  char *str = strdup(orig);
  char *s = str;
  int i;
  int n_groups = count_class_groups(str);
  self->class_groups = realloc_or_die(self->class_groups,
      (n_groups + 2) * sizeof(ClassifyClassGroup));
  for (i = 0; i < n_groups; i++){
    ClassifyClassGroup *group = &self->class_groups[i];
    group->classes = s;
    group->n_classes = 0;
    group->offset = s - str;
    for (; *s && *s != ','; s++){
      group->n_classes++;
    }
    s++;
    GST_LOG("group %d has %d classes", i, group->n_classes);
  }
  self->n_groups = n_groups;
  return s - str - 1;
}

#define CLASSIFY_METADATA_DEFAULTS(self) {                              \
    PP_GET_STRING(self, PROP_CLASSES, DEFAULT_PROP_CLASSES),            \
    PP_GET_FLOAT(self, PROP_MIN_FREQUENCY, DEFAULT_MIN_FREQUENCY),    \
    PP_GET_FLOAT(self, PROP_MAX_FREQUENCY, DEFAULT_MAX_FREQUENCY),    \
    PP_GET_FLOAT(self, PROP_KNEE_FREQUENCY, DEFAULT_KNEE_FREQUENCY),  \
    PP_GET_INT(self, PROP_MFCCS, DEFAULT_PROP_MFCCS),                   \
    PP_GET_INT(self, PROP_WINDOW_SIZE, DEFAULT_WINDOW_SIZE),            \
    PP_GET_STRING(self, PROP_BASENAME, DEFAULT_BASENAME),               \
    PP_GET_INT(self, PROP_DELTA_FEATURES, DEFAULT_PROP_DELTA_FEATURES), \
    PP_GET_FLOAT(self, PROP_FOCUS_FREQUENCY, DEFAULT_FOCUS_FREQUENCY),  \
    PP_GET_FLOAT(self, PROP_LAG, DEFAULT_PROP_LAG),                     \
    PP_GET_BOOLEAN(self, PROP_INTENSITY_FEATURE, DEFAULT_PROP_INTENSITY_FEATURE), \
    PP_GET_FLOAT(self, PROP_CONFIRMATION_LAG, DEFAULT_PROP_CONFIRMATION_LAG), \
    }

static char*
construct_metadata(GstClassify *self, struct ClassifyMetadata *m){
  char *metadata;
  struct ClassifyMetadata m_defaults = CLASSIFY_METADATA_DEFAULTS(self);
  if (m == NULL){
    m = &m_defaults;
  }
  int ret = asprintf(&metadata,
      "classes %s\n"
      "min-frequency %f\n"
      "max-frequency %f\n"
      "knee-frequency %f\n"
      "mfccs %d\n"
      "window-size %d\n"
      "basename %s\n"
      "delta-features %d\n"
      "focus-frequency %f\n"
      "lag %f\n"
      "intensity-feature %d\n"
      "confirmation-lag %f\n"
      ,
      m->classes,
      m->min_freq,
      m->max_freq,
      m->knee_freq,
      m->mfccs,
      m->window_size,
      m->basename,
      m->delta_features,
      m->focus_freq,
      m->lag,
      m->intensity_feature,
      m->confirmation_lag
  );
  STDERR_DEBUG("%s", metadata);
  if (ret == -1){
    FATAL_ERROR("can't alloc memory for metadata. or something.");
  }
  return metadata;
}

static int
load_metadata(const char *metadata, struct ClassifyMetadata *m){
  if (! metadata){
    GST_WARNING("There is no metadata!");
    return -1;
  }
  /*New metadata items always need to added at the bottom, even if it would
    make more sense to have them elsewhere -- so that old nets can properly be
    loaded.

    XXX alternately, there might be something better than sscanf().
   */
  const int expected_n = 11;
  const char *template = (
      "classes %ms "
      "min-frequency %f "
      "max-frequency %f "
      "knee-frequency %f "
      "mfccs %d "
      "window-size %d "
      "basename %ms "
      "delta-features %d "
      "focus-frequency %f "
      "lag %f "
      "intensity-feature %d"
      "confirmation-lag %f"
  );
  int n = sscanf(metadata, template, &m->classes,
      &m->min_freq, &m->max_freq, &m->knee_freq,
      &m->mfccs, &m->window_size, &m->basename, &m->delta_features,
      &m->focus_freq, &m->lag, &m->intensity_feature,
      &m->confirmation_lag);
  if (n != expected_n){
    GST_WARNING("Found only %d/%d metadata items", n, expected_n);
  }
  return expected_n - n;
}

static void
free_metadata_items(struct ClassifyMetadata *m){
  /*these are const char*, but free complains about the const */
  free((char *)m->classes);
  m->classes = NULL;
  free((char *)m->basename);
  m->basename = NULL;
}

static void
setup_audio(GstClassify *self, int window_size, int mfccs, float min_freq,
    float max_freq, float knee_freq, float focus_freq, int delta_features,
    int intensity_feature, float lag, float confirmation_lag){
  /*List arguments to help make sure they have been passed in in the right
    order!*/
  GST_DEBUG("setting up audio thus:\n"
      "window_size %d\n"
      "mfccs %d\n"
      "min_freq %f\n"
      "max_freq %f\n"
      "knee_freq %f\n"
      "focus_freq %f\n"
      "delta_features %d\n"
      "intensity_feature %d\n"
      "lag %f\n"
      "confirmation_lag %f\n",
      window_size, mfccs, min_freq,
      max_freq, knee_freq, focus_freq, delta_features,
      intensity_feature, lag, confirmation_lag);

  self->mfcc_factory = recur_audio_binner_new(window_size,
      RECUR_WINDOW_HANN,
      CLASSIFY_N_FFT_BINS,
      min_freq, max_freq, knee_freq, focus_freq,
      CLASSIFY_RATE,
      1.0f / 32768,
      CLASSIFY_VALUE_SIZE);

  self->window_size = window_size;
  self->delta_features = delta_features;
  self->intensity_feature = intensity_feature ? 1 : 0;
  self->lag = lag;
  self->confirmation_lag = confirmation_lag;
  GST_LOG("mfccs: %d", mfccs);
  self->mfccs = mfccs;
}

static RecurNN *
load_specified_net(GstClassify *self, const char *filename){
  struct ClassifyMetadata m = CLASSIFY_METADATA_DEFAULTS(self);
  int force_load = PP_GET_BOOLEAN(self, PROP_FORCE_LOAD, DEFAULT_PROP_FORCE_LOAD);
  RecurNN *net = rnn_load_net(filename);
  if (net == NULL){
    FATAL_ERROR("Could not load %s", filename);
  }
  GST_DEBUG("loaded metadata: %s", net->metadata);

  int unloaded_items = load_metadata(net->metadata, &m);
  if (unloaded_items){
    if (force_load){
      STDERR_DEBUG("continuing despite metadata mismatch (%d not loaded), "
          "because force-load is set\n%s", unloaded_items, net->metadata);
      free(net->metadata);
      net->metadata = construct_metadata(self, &m);
      STDERR_DEBUG("New metadata:\n"
          "%s", net->metadata);
    }
    else {
      FATAL_ERROR("The metadata (%s) is bad", net->metadata);
    }
  }
  int n_outputs = parse_classes_string(self, m.classes);
  if (n_outputs != net->output_size){
    FATAL_ERROR("Class '%s' string suggests %d outputs, net has %d",
        m.classes, n_outputs, net->output_size);
  }
  if (self->mfcc_factory != NULL ||
      self->net != NULL){
    FATAL_ERROR("There is already a net (%p) and/or audiobinner (%p). "
        "This won't work", self->net, self->mfcc_factory);
  }
  self->net_filename = strdup(filename);
  self->basename = strdup(m.basename);
  setup_audio(self, m.window_size, m.mfccs, m.min_freq,
      m.max_freq, m.knee_freq, m.focus_freq, m.delta_features,
      m.intensity_feature, m.lag, m.confirmation_lag);
  self->net = net;
  if (! unloaded_items){
    /* in the unloaded_items case, it might be that some of the strings that
       would need freeing were not actually allocated. It is best to just let
       them leak (they are short and this should be rare).
     */
    free_metadata_items(&m);
  }
  return net;
}

static inline void
initialise_net(GstClassify *self, RecurNN *net)
{
  /*start off with a default set of parameters */
  struct RecurInitialisationParameters p;

  rnn_init_default_weight_parameters(net, &p);
  /*if the init-method is not set, guess based on related properties (for
    back-compatibility and possibly DWIMery).*/
  if (PP_IS_SET(self, PROP_WEIGHT_INIT_METHOD)){
    p.method = PP_GET_INT(self, PROP_WEIGHT_INIT_METHOD, p.method);
  }
  else if (PP_IS_SET(self, PROP_WEIGHT_FAN_IN_SUM)){
    p.method = RNN_INIT_FAN_IN;
  }

  p.fan_in_sum = PP_GET_FLOAT(self, PROP_WEIGHT_FAN_IN_SUM, p.fan_in_sum);
  p.fan_in_step = PP_GET_FLOAT(self, PROP_WEIGHT_FAN_IN_KURTOSIS,
      p.fan_in_step);

  rnn_randomise_weights_clever(net, &p);
}

static RecurNN *
create_net(GstClassify *self, int bottom_layer_size,
    int hidden_size, int top_layer_size, char *metadata){
  if (self->mfcc_factory == NULL){
    FATAL_ERROR("We seem to be creating a net before the audio stuff"
        " has been set up. It won't work.");
  }
  RecurNN *net;
  int n_features = get_n_features(self);
  u32 flags = CLASSIFY_RNN_FLAGS;
  int bptt_depth = PP_GET_INT(self, PROP_BPTT_DEPTH, DEFAULT_PROP_BPTT_DEPTH);
  float momentum = PP_GET_FLOAT(self, PROP_MOMENTUM, DEFAULT_PROP_MOMENTUM);
  float learn_rate = PP_GET_FLOAT(self, PROP_LEARN_RATE, DEFAULT_LEARN_RATE);
  float bottom_learn_rate_scale = PP_GET_FLOAT(self, PROP_BOTTOM_LEARN_RATE_SCALE,
      DEFAULT_BOTTOM_LEARN_RATE_SCALE);
  float top_learn_rate_scale = PP_GET_FLOAT(self, PROP_TOP_LEARN_RATE_SCALE,
      DEFAULT_TOP_LEARN_RATE_SCALE);
  u64 rng_seed = get_gvalue_u64(PENDING_PROP(self, PROP_RNG_SEED), DEFAULT_RNG_SEED);
  GST_DEBUG("rng seed %lu", rng_seed);
  float weight_init_scale = PP_GET_FLOAT(self, PROP_WEIGHT_INIT_SCALE,
    DEFAULT_PROP_WEIGHT_INIT_SCALE);
  float presynaptic_noise = PP_GET_FLOAT(self, PROP_PRESYNAPTIC_NOISE,
    DEFAULT_PROP_PRESYNAPTIC_NOISE);

  int lawnmower = PP_GET_BOOLEAN(self, PROP_LAWN_MOWER, DEFAULT_PROP_LAWN_MOWER);
  if (lawnmower){
    flags |= RNN_COND_USE_LAWN_MOWER;
  }
  else {
    flags &= ~RNN_COND_USE_LAWN_MOWER;
  }
  net = rnn_new_with_bottom_layer(n_features, bottom_layer_size, hidden_size,
      top_layer_size, flags, rng_seed,
      NULL, bptt_depth, learn_rate, momentum, presynaptic_noise, RNN_RELU, 0);

  initialise_net(self, net);

  if (self->learning_style == RNN_ADAGRAD){
    float ada_ballast = PP_GET_FLOAT(self, PROP_ADAGRAD_BALLAST,
        DEFAULT_PROP_ADAGRAD_BALLAST);
    rnn_set_momentum_values(net, ada_ballast);
  }

  if (weight_init_scale){
    rnn_scale_initial_weights(net, weight_init_scale);
  }
  net->bptt->ho_scale = top_learn_rate_scale;
  if (net->bottom_layer){
    net->bottom_layer->learn_rate_scale = bottom_learn_rate_scale;
  }
  if (PERIODIC_PGM_DUMP){
    rnn_multi_pgm_dump(net, PERIODIC_PGM_DUMPEES, self->basename);
  }
  net->metadata = strdup(metadata);
  return net;
}


static RecurNN *
load_or_create_net(GstClassify *self){
  char *metadata = construct_metadata(self, NULL);
  int hidden_size = PP_GET_INT(self, PROP_HIDDEN_SIZE, DEFAULT_HIDDEN_SIZE);
  int bottom_layer_size = PP_GET_INT(self, PROP_BOTTOM_LAYER, 0);
  const char *class_string = PP_GET_STRING(self, PROP_CLASSES, DEFAULT_PROP_CLASSES);
  int top_layer_size = count_class_group_members(class_string);
  int force_load = PP_GET_BOOLEAN(self, PROP_FORCE_LOAD, DEFAULT_PROP_FORCE_LOAD);
  if (self->net_filename == NULL){
    set_net_filename(self, hidden_size, bottom_layer_size, top_layer_size, metadata);
  }
  RecurNN *net = TRY_RELOAD ? rnn_load_net(self->net_filename) : NULL;
  if (net){
    if (net->output_size != top_layer_size ||
        net->hidden_size != hidden_size ||
        (net->bottom_layer && ! bottom_layer_size) ||
        (net->metadata && ! force_load && strcmp(net->metadata, metadata))){
      FATAL_ERROR("I thought I could load the file '%s',\n"
          "but it doesn't seem to match the layer sizes and metadata I want.\n"
          "If you mean to continue with a freshly made net, please move\n"
          "that file aside. If you are sure you mean to use that file,\n"
          "specify it directly using the 'net-filename' property. If you\n"
          "are trying to do domething like use the same net with different\n"
          "audio metadata, then you are out of luck, for now at least. Sorry. Most\n"
          "likely any problems are my fault. The compared layer sizes are:\n"
          "output: expected %d,  loaded %d\n"
          "hidden: expected %d,  loaded %d\n"
          "bottom: expected %d,  loaded %d\n"
          "and the metadata is:\n"
          "expected:\n%s\n"
          "loaded:\n%s\n",
          self->net_filename,
          top_layer_size, net->output_size,
          hidden_size, net->hidden_size,
          bottom_layer_size, net->bottom_layer ? net->bottom_layer->output_size : 0,
          metadata, net->metadata);
    }
    /*XXX need to get the likes of PROP_LEARN_RATE, PROP_TOP_LEARN_RATE_SCALE,
      PROP_MOMENTUM, PROP_BOTTOM_LEARN_RATE_SCALE from gvalue store.
    */
  }
  else {
    net = create_net(self, bottom_layer_size, hidden_size, top_layer_size, metadata);
  }
  struct ClassifyMetadata m = {0};
  int unloaded_items = load_metadata(net->metadata, &m);
  int n_outputs = parse_classes_string(self, m.classes);
  if (n_outputs != net->output_size ||
      strcmp(class_string, m.classes)){
    FATAL_ERROR("Metadata class string %s suggests %d outputs, net has %s and %d",
        m.classes, n_outputs, class_string, net->output_size);
  }
  if (! unloaded_items){
    free_metadata_items(&m);
  }
  free(metadata);
  return net;
}


static inline void
reset_channel_targets(GstClassify *self){
  int i, j;
  /*out of bounds [0, n_channels - 1) signals no target */
  for (i = 0; i < self->n_channels; i++){
    for (j = 0; j < self->n_groups; j++){
      self->channels[i].group_target[j] = -1;
    }
  }
}

static void
load_or_create_net_and_audio(GstClassify *self)
{
  /*there are two paths to loading a net.

      1. If the net has been specifically named in the 'net-filename'
      property, it (and the audio feature extraction) will have been set up
      already from the set_property hook.

      2. If no net name is specified, the net is created here. First the audio
      parameters and layer sizes are determined, then a network name is
      created from them and the net made.
  */
  if (self->mfcc_factory != NULL){
    FATAL_ERROR("mfcc_factory exists before net. This won't work.");
  }
  self->basename = strdup(PP_GET_STRING(self, PROP_BASENAME, DEFAULT_BASENAME));
  setup_audio(self, PP_GET_INT(self, PROP_WINDOW_SIZE, DEFAULT_WINDOW_SIZE),
      PP_GET_INT(self, PROP_MFCCS, DEFAULT_PROP_MFCCS),
      PP_GET_FLOAT(self, PROP_MIN_FREQUENCY, DEFAULT_MIN_FREQUENCY),
      PP_GET_FLOAT(self, PROP_MAX_FREQUENCY, DEFAULT_MAX_FREQUENCY),
      PP_GET_FLOAT(self, PROP_KNEE_FREQUENCY, DEFAULT_KNEE_FREQUENCY),
      PP_GET_FLOAT(self, PROP_FOCUS_FREQUENCY, DEFAULT_FOCUS_FREQUENCY),
      PP_GET_INT(self, PROP_DELTA_FEATURES, DEFAULT_PROP_DELTA_FEATURES),
      PP_GET_BOOLEAN(self, PROP_INTENSITY_FEATURE, DEFAULT_PROP_INTENSITY_FEATURE),
      PP_GET_FLOAT(self, PROP_LAG, DEFAULT_PROP_LAG),
      PP_GET_FLOAT(self, PROP_CONFIRMATION_LAG, DEFAULT_PROP_CONFIRMATION_LAG)
  );
  self->net = load_or_create_net(self);

  RecurNN *net = self->net;
  if (net->bottom_layer && 0){
    self->error_image = temporal_ppm_alloc(net->bottom_layer->o_size, 300,
        "bottom_error", 0, PGM_DUMP_COLOUR, &net->bottom_layer->o_error);
  }
}

/*start() is called once by gstreamer just before the first call to setup()
  below */
static gboolean
start(GstBaseTransform *trans){
  GstClassify *self = GST_CLASSIFY(trans);
  if (self->net == NULL){
    load_or_create_net_and_audio(self);
  }
  return TRUE;
}

/*gst_classify_setup is called every time the pipeline starts up -- that is,
  for every new set of input files.
 */
static gboolean
gst_classify_setup(GstAudioFilter *base, const GstAudioInfo *info){
  GST_INFO("gst_classify_setup\n");
  GstClassify *self = GST_CLASSIFY(base);

  if (self->n_channels != info->channels){
    DEBUG("given %d channels, previous %d", info->channels, self->n_channels);
    if (self->audio_queue){
      free(self->audio_queue);
    }
    self->n_channels = info->channels;
    self->queue_size = info->channels * self->window_size * CLASSIFY_QUEUE_FACTOR;
    int alloc_size = self->queue_size + info->channels * self->window_size / 2;
    self->audio_queue = malloc_aligned_or_die(alloc_size * sizeof(s16));
    if (self->channels){
      for (int i = 0; i < self->n_channels; i++){
        finalise_channel(&self->channels[i]);
      }
      free(self->channels);
    }
    self->channels = malloc_aligned_or_die(self->n_channels * sizeof(ClassifyChannel));
    if (self->subnets){
      free(self->subnets);
    }
    self->subnets = rnn_new_training_set(self->net, self->n_channels);
    for (int i = 0; i < self->n_channels; i++){
      init_channel(&self->channels[i], self->subnets[i],
          self->window_size, i, self->n_groups, self->delta_features);
    }
  }
  const float ignore_start = PP_GET_FLOAT(self, PROP_IGNORE_START,
      DEFAULT_PROP_IGNORE_START);
  self->ignored_windows = (int)(ignore_start * CLASSIFY_RATE *
          2.0f / self->window_size + 0.5f);
  if (self->ignored_windows){
    GST_DEBUG("ignoring times less than %f (window_no %d)",
        ignore_start, self->ignored_windows);
  }

  reset_channel_targets(self);
  maybe_start_logging(self);
  maybe_parse_target_string(self);
  maybe_parse_error_weight_string(self);

  GstStructure *s = gst_structure_new_empty("classify-setup");
  GstMessage *msg = gst_message_new_element(GST_OBJECT(self), s);
  gst_element_post_message(GST_ELEMENT(self), msg);

  /*A risk with this as written is that little bits of audio might end up
    getting left behind or overwritten*/
  if (self->random_alignment && self->training){
    self->write_offset = 0;
    int offset = rand_small_int(&self->net->rng, self->window_size) - self->window_size / 2;
    self->read_offset = offset * self->n_channels;
    if (offset < 0){
      self->read_offset += self->queue_size;
      memset(self->audio_queue + self->read_offset, 0,
          (self->queue_size - self->read_offset) * sizeof(s16));
    }
    GST_LOG("random offset is %d (%d * %d)",
        self->read_offset, offset, self->n_channels);
  }
  else {//XXX?
    self->write_offset = 0;
    self->read_offset = 0;
  }
  if (self->training && self->weight_noise){
    float wn = self->weight_noise;
    if (wn < 0){
      wn *= -self->net->bptt->learn_rate;
    }
    rnn_weight_noise(self->net, wn);
  }

  GST_DEBUG("finished setup()");
  return TRUE;
}

static int
cmp_class_event(const void *a, const void *b){
  const int at = ((const ClassifyClassEvent *)a)->window_no;
  const int bt = ((const ClassifyClassEvent *)b)->window_no;
  return at - bt;
}

/*target specification.
  channel    := 'c'<int>
  time       := 't'<float>
  target     := <letter> | '=' | '-'
  targets    := <target>+
  event      := <class><time>':'<targets>
  space      := ' '
  target_string := (<event><space>)*<event>

  The number of targets must equal the number of exclusive categorical groups,
  and the target characters must match the class characters, or be either of
  the two special characters. '=' means the group target is not being changed,
  while '-' means there is no target for that group.

  For example, if the classes had been set to "Mm,Kk,Wx", this would be a
  valid target string:

  "c0t2.3:mkW c0t4.1:m-x c11t123:=K= c1t0:Mkx"

  All channels and categories start off in an untargetted state, as if they
  had been passed the '-' token.
 */
static inline int
parse_complex_target_string(GstClassify *self, const char *str){
  char *e;
  const char *s;
  int i;
  const float time_to_window_no = CLASSIFY_RATE * 2.0f / self->window_size;

#define TIME_TO_WINDOW_NO(x) (((x) + self->lag) * time_to_window_no + 0.5)
#define WINDOW_NO_TO_TIME(x) ((x) / time_to_window_no - self->lag)

  ClassifyClassEvent *ev;
  /* the number of characters in the string between a colon and a space,
     excluding equals signs, is the number of events.*/
  int n_events = 0;
  int n_phrases = 0;
  int counting = 0;
  for (s = str; *s; s++){
    if (*s == ':'){
      n_phrases++;
      counting = 1;
    }
    else if (*s == ' '){
      counting = 0;
    }
    else if (counting){
      n_events += (*s != '=');
    }
  }
  if (self->n_class_events < n_events){
    GST_DEBUG("found %d targets events, up from %d. Reallocing self->class_events (%p)",
        n_events, self->n_class_events, self->class_events);
    self->class_events = realloc_or_die(self->class_events,
        (n_events + 1) * sizeof(ClassifyClassEvent));
  }
  GST_DEBUG("events %p, n %d", self->class_events, n_events);

  s = str;

  ev = self->class_events;
  for (i = 0; i < n_phrases; i++){
#define ERROR_IF(x) if (x) goto parse_error
    int channel;
    int window_no;

    /*  channel    := 'c'<int>  */
    GST_DEBUG("looking for 'c', got '%c'", *s);
    ERROR_IF(*s != 'c');
    s++;
    channel = strtol(s, &e, 10);
    ERROR_IF(s == e);
    ERROR_IF(channel < 0 || channel >= self->n_channels);
    GST_DEBUG("channel %d", channel);
    /*  time       := 't'<float> */
    ERROR_IF(*e != 't');
    GST_DEBUG("looking for 't', got '%c'", *e);
    s = e + 1;
    float time = strtod(s, &e);
    window_no = TIME_TO_WINDOW_NO(time);
    GST_DEBUG("time %f window_no %d", time, window_no);

    ERROR_IF(s == e || window_no < 0);

    /* ':' */
    ERROR_IF(*e != ':');
    s = e + 1;
    for (int j = 0; j < self->n_groups; j++){
      /*  target     := <letter> | '=' | '-' */
      ClassifyClassGroup *g = &self->class_groups[j];
      GST_DEBUG("looking at letter '%c'", *s);
      if (*s == '-'){
        /* no training: set target to -1*/
        ev->channel = channel;
        ev->class_group = j;
        ev->window_no = window_no;
        ev->target = -1;
        ev++;
      }
      else if (*s != '='){
        int k;
        /* set training target to index of this character */
        GST_DEBUG("found a letter");

        for (k = 0; k < g->n_classes; k++){
          if (g->classes[k] == *s){
            ev->channel = channel;
            ev->class_group = j;
            ev->window_no = window_no;
            ev->target = k;
            ev++;
            break;
          }
        }
        ERROR_IF(k == g->n_classes);
      }
      /*the '=' case means no change for this group, so no event */
      s++;
    }
    //s = e;
    if (*s != ' '){
      if (i != n_phrases - 1){
        GST_DEBUG("event list ended peculiarly early (%d != %d - 1)", i, n_phrases);
      }
      break;
    }
    s++;
    GST_LOG("event: channel %d target %d window %d starting %.2f (request %.2f)",
        ev->channel, ev->target, ev->window_no,
        (double)WINDOW_NO_TO_TIME(ev->window_no), time);
#undef ERROR_IF
  }
  qsort(self->class_events, n_events, sizeof(ClassifyClassEvent), cmp_class_event);
  self->n_class_events = n_events;
  self->class_events_index = 0;
#if 0
  for (i = 0; i < n_events; i++){
    ev = &self->class_events[i];
    fprintf(stderr, "c%dt%.2f:%c ",
        ev->channel, WINDOW_NO_TO_TIME(ev->window_no),
        self->class_groups[0].classes[ev->target]);
  }
  fprintf(stderr, "\n");
#endif
  return 0;
 parse_error:
  GST_ERROR("Can't parse '%s' into %d events for %d channels: "
      "stopping after %d events (%ld chars)",
      str, n_events, self->n_channels, i, s - str);
  self->n_class_events = 0;
  return -1;

#undef TIME_TO_WINDOW_NO
#undef WINDOW_NO_TO_TIME
}

static void
maybe_parse_target_string(GstClassify *self){
  if (self->channels == NULL){
    GST_DEBUG("not parsing target string because channels is NULL");
    return;
  }
  char *target_string = steal_gvalue_string(PENDING_PROP(self, PROP_TARGET));
  char *s = target_string;
  if (s == NULL){
    GST_DEBUG("not parsing NULL target string");
    return;
  }
  GST_DEBUG("parsing target '%s'", s);
  if (*s == 0){
    reset_channel_targets(self);
  }
  else if (*s == 'c'){
    parse_complex_target_string(self, s);
  }
  else {
    GST_DEBUG("simple training mode is GONE!");
  }
  free(target_string);
  self->window_no = 0;
}

static void
maybe_parse_error_weight_string(GstClassify *self){
  /*the error weight string typically looks something like this "5:3:4", which
    gives the first class 5/3 the weight of the second and 5/4 that of the
    third, by multiplying each weight by the corresponding number. To maintain
    the overall error scale, you would use "1.25:0.75:1" for the same weight
    ratio.

    The syntax is rather loose -- any non-numeric character will do as a
    separator. This means, depending on locale, it happens to work if you have
    multiple groups separated by commas (e.g. "1:1:3,2:1"). That will probably
    fail in locales where the comma is the decimal point, and it is probably
    wiser to continue with colons or something else (e.g "1:1:3 2:1"). At some
    point the syntax might get stricter.

    In the two class case, this may have little effect other than scaling the
    overall error.
  */

  char *orig, *e, *s;
  int i;
  if (self->channels == NULL){
    GST_DEBUG("not parsing error_weight string because channels is NULL");
    return;
  }
  e = orig = s = steal_gvalue_string(PENDING_PROP(self, PROP_ERROR_WEIGHT));
  if (s == NULL){
    GST_DEBUG("not parsing error_weight string because it is  NULL");
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
  const char *s = get_gvalue_string(PENDING_PROP(self, PROP_LOG_FILE), NULL);
  GST_DEBUG("pending log '%s'; subnets is %p", s, self->subnets);
  if (s && self->subnets){
    if (s[0] == 0){
      rnn_set_log_file(self->net, NULL, 0);
    }
    else {
      rnn_set_log_file(self->net, s, 1);
    }
    g_value_unset(PENDING_PROP(self, PROP_LOG_FILE));
  }
}

static void
maybe_set_net_scalar(GstClassify *self, guint prop_id, const GValue *value)
{

#define SET_FLOAT(var) do {var = get_gvalue_float(value, var);} while(0)

  RecurNN *net = self->net;
  if (net){
    switch (prop_id){
    case PROP_LEARN_RATE:
      SET_FLOAT(net->bptt->learn_rate);
      break;
    case PROP_TOP_LEARN_RATE_SCALE:
      SET_FLOAT(net->bptt->ho_scale);
      break;
    case PROP_MOMENTUM:
      SET_FLOAT(net->bptt->momentum);
      break;
    case PROP_PRESYNAPTIC_NOISE:
      SET_FLOAT(net->presynaptic_noise);
      break;
    case PROP_BOTTOM_LEARN_RATE_SCALE:
      if (net->bottom_layer){
        SET_FLOAT(net->bottom_layer->learn_rate_scale);
      }
      break;
    }
  }
  else {
    copy_gvalue(PENDING_PROP(self, prop_id), value);
  }

#undef SET_FLOAT

}

static void
gst_classify_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstClassify *self = GST_CLASSIFY (object);
  GST_DEBUG("gst_classify_set_property with prop_id %d\n", prop_id);
  if (value){
    switch (prop_id) {
    case PROP_PGM_DUMP:
      if (self->net){
        const char *s = g_value_get_string(value);
        rnn_multi_pgm_dump(self->net, s, self->basename);
      }
      break;

    case PROP_SAVE_NET:
      if (self->net){
        const char *s = g_value_get_string(value);
        if (s && s[0] != 0){
          rnn_save_net(self->net, s, 1);
        }
        else {
          rnn_save_net(self->net, self->net_filename, 1);
        }
      }
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

    case PROP_RANDOM_ALIGNMENT:
      self->random_alignment = g_value_get_boolean(value);

    case PROP_TRAINING:
      self->training = g_value_get_boolean(value);
      break;

    case PROP_MOMENTUM_SOFT_START:
      self->momentum_soft_start = g_value_get_float(value);
      break;

    case PROP_WEIGHT_NOISE:
      self->weight_noise = g_value_get_float(value);
      break;

    case PROP_LEARNING_STYLE:
      self->learning_style = g_value_get_int(value);
      break;

      /*this causes the net to be loaded or created based on the properties so
        far.*/
    case PROP_LOAD_NET_NOW:
      if (self->net == NULL){
        load_or_create_net_and_audio(self);
      }
      else {
        GST_WARNING("There is a net already. Let's not make a new one");
      }
      /*this causes the net to be loaded immediately, if possible, so that
        various properties can be queried */
    case PROP_NET_FILENAME:
      if (self->net == NULL){
        const char *s = g_value_get_string(value);
        load_specified_net(self, s);
        //XXX what to do on error
      }
      else {
        GST_WARNING("it is TOO LATE to set %s.", pspec->name);
      }
      break;

      /*These ones affect the net directly if the net exists. Otherwise they
        need to be stored as pending properties.*/
    case PROP_TOP_LEARN_RATE_SCALE:
    case PROP_BOTTOM_LEARN_RATE_SCALE:
    case PROP_LEARN_RATE:
    case PROP_PRESYNAPTIC_NOISE:
    case PROP_MOMENTUM:
      maybe_set_net_scalar(self, prop_id, value);
      break;
    /*These properties only need to be stored until net creation, and can't
      be changed afterwards.
    */
    case PROP_FORCE_LOAD:
    case PROP_BASENAME:
    case PROP_LAG:
    case PROP_CONFIRMATION_LAG:
    case PROP_MIN_FREQUENCY:
    case PROP_KNEE_FREQUENCY:
    case PROP_FOCUS_FREQUENCY:
    case PROP_MAX_FREQUENCY:
    case PROP_CLASSES:
    case PROP_BOTTOM_LAYER:
    case PROP_HIDDEN_SIZE:
    case PROP_BPTT_DEPTH:
    case PROP_LAWN_MOWER:
    case PROP_WEIGHT_INIT_METHOD:
    case PROP_WEIGHT_FAN_IN_SUM:
    case PROP_WEIGHT_FAN_IN_KURTOSIS:
    case PROP_RNG_SEED:
    case PROP_WINDOW_SIZE:
    case PROP_DELTA_FEATURES:
    case PROP_INTENSITY_FEATURE:
    case PROP_MFCCS:
    case PROP_WEIGHT_INIT_SCALE:
      if (self->net == NULL){
        copy_gvalue(PENDING_PROP(self, prop_id), value);
      }
      else {
        GST_WARNING("it is TOO LATE to set %s.", pspec->name);
      }
      break;
    /*these ones can be set any time but only have effect in
      gst_classify_setup(). that is, after a net exists but possibly more than
      once.*/
    case PROP_IGNORE_START:
    case PROP_LOG_FILE:
    case PROP_TARGET:
    case PROP_ERROR_WEIGHT:
      copy_gvalue(PENDING_PROP(self, prop_id), value);
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
  RecurNN *net = self->net;

  switch (prop_id) {

#define NET_OR_PP(var, type, needs_bptt, needs_bottom_layer) do {       \
    if (net && (!(needs_bptt) || net->bptt) &&                          \
        (!(needs_bottom_layer) || net->bottom_layer)){                  \
      g_value_set_ ## type(value, var);                                 \
    } else {                                                            \
      const GValue *pp = PENDING_PROP(self, prop_id);                   \
      if (G_IS_VALUE(pp)){                                              \
        g_value_copy(pp, value);                                        \
      }                                                                 \
    }} while(0)

  case PROP_LEARN_RATE:
    NET_OR_PP(net->bptt->learn_rate, float, 1, 0);
    break;
  case PROP_TOP_LEARN_RATE_SCALE:
    NET_OR_PP(net->bptt->ho_scale, float, 1, 0);
    break;
  case PROP_MOMENTUM:
    NET_OR_PP(net->bptt->momentum, float, 1, 0);
    break;
  case PROP_BOTTOM_LEARN_RATE_SCALE:
    NET_OR_PP(net->bottom_layer->learn_rate_scale, float, 0, 1);
    break;
  case PROP_BPTT_DEPTH:
    NET_OR_PP(net->bptt->depth, int, 1, 0);
    break;
  case PROP_HIDDEN_SIZE:
    NET_OR_PP(net->hidden_size, int, 0, 0);
    break;
  case PROP_BOTTOM_LAYER:
    NET_OR_PP(net->bottom_layer->input_size, int, 0, 1);
    break;
  case PROP_LAWN_MOWER:
    NET_OR_PP(net->flags & RNN_COND_USE_LAWN_MOWER, boolean, 0, 0);
    break;
  case PROP_BASENAME:
    NET_OR_PP(self->basename, string, 0, 0);
    break;
  case PROP_CLASSES:
    NET_OR_PP(self->class_groups[0].classes, string, 0, 0);
    break;
  case PROP_NET_FILENAME:
    NET_OR_PP(self->net_filename, string, 0, 0);
    break;



#undef NET_OR_PP

  case PROP_GENERATION:
    if (net){
      g_value_set_uint(value, net->generation);
    }
    else {
      g_value_set_uint(value, 0);
    }
    break;
  case PROP_TRAINING:
    g_value_set_boolean(value, self->training);
    break;
  case PROP_MFCCS:
    g_value_set_int(value, self->mfccs);
    break;
  case PROP_MOMENTUM_SOFT_START:
    g_value_set_float(value, self->momentum_soft_start);
    break;
  case PROP_WEIGHT_NOISE:
    g_value_set_float(value, self->weight_noise);
    break;
  case PROP_LEARNING_STYLE:
    g_value_set_int(value, self->learning_style);
    break;
  case PROP_WINDOW_SIZE:
    g_value_set_int(value, self->window_size);
    break;

  case PROP_WINDOWS_PER_SECOND:
    g_value_set_double(value, CLASSIFY_RATE * 2.0f / self->window_size);
  break;

  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    break;
  }
}

/*XXX shared with parrot and possibly others */
static inline void
possibly_save_net(RecurNN *net, const char *filename)
{
  GST_LOG("possibly saving to %s", filename);
  if (PERIODIC_SAVE_NET && (net->generation & 511) == 0){
    rnn_save_net(net, filename, 1);
  }
}


static inline void
send_message(GstClassify *self, float mean_err, float now, GstClockTime pts)
{
  GstMessage *msg;
  char *name = "classify";
  GstStructure *s;
  s = gst_structure_new (name,
      "error", G_TYPE_FLOAT, mean_err,
      "time", G_TYPE_FLOAT, now,
      NULL);
  for (int i = 0; i < self->n_channels; i++){
    char key[60];
    char id[40];
    ClassifyChannel *c = &self->channels[i];
    RecurNN *net = c->net;
    for (int j = 0; j < self->n_groups; j++){
      ClassifyClassGroup *g = &self->class_groups[j];
      int target = c->group_target[j];
      int winner = c->group_winner[j];
      snprintf(id, sizeof(id), "channel %d, group %d", i, j);
      for (int k = 0; k < g->n_classes; k++){
        int t = g->classes[k];
        float error = net->bptt->o_error[g->offset + k];
        snprintf(key, sizeof(key), "%s %c", id, t);
        gst_structure_set(s, key, G_TYPE_FLOAT, -error, NULL);
      }
      if (target >= 0 && target < g->n_classes){
        char t_char = g->classes[target];
        snprintf(key, sizeof(key), "%s correct", id);
        gst_structure_set(s, key, G_TYPE_INT, winner == target, NULL);
        snprintf(key, sizeof(key), "%s target", id);
        gst_structure_set(s, key, G_TYPE_CHAR, t_char, NULL);
      }
      if (winner >= 0 && winner < g->n_classes){
        char w_char = g->classes[winner];
        snprintf(key, sizeof(key), "%s winner", id);
        gst_structure_set(s, key, G_TYPE_CHAR, w_char, NULL);
      }
    }
  }
  msg = gst_message_new_element(GST_OBJECT(self), s);
  msg->timestamp = pts;
  gst_element_post_message(GST_ELEMENT(self), msg);
}


static inline void
pcm_to_features(RecurAudioBinner *mf, ClassifyChannel *c, int mfccs,
    int delta_features, int intensity_feature){
  float *pcm = c->pcm_now;
  float *answer;
  int n_raw_features;
  if (mfccs){
    answer = recur_extract_mfccs(mf, pcm) + 1 - intensity_feature;
    n_raw_features = mfccs + intensity_feature;
  }
  else {
    answer = recur_extract_log_freq_bins(mf, pcm);
    n_raw_features = CLASSIFY_N_FFT_BINS + intensity_feature;
    if (intensity_feature){
      float sum = 0;
      for (int i = 0; i < CLASSIFY_N_FFT_BINS; i++){
        sum += answer[i];
      }
      answer[CLASSIFY_N_FFT_BINS] = sum / CLASSIFY_N_FFT_BINS;
    }
  }
  if (c->prev_features){
    float *tmp = c->features;
    c->features = c->prev_features;
    c->prev_features = tmp;
  }
  for (int i = 0; i < n_raw_features; i++){
    c->features[i] = answer[i];
  }
  if (c->prev_features){
    for (int j = (delta_features + 1) * n_raw_features - 1;
         j >= n_raw_features; j--){
      int i = j - n_raw_features;
      c->features[j] = c->features[i] - c->prev_features[i];
    }
  }
}

static inline ClassifyChannel *
prepare_channel_features(GstClassify *self, s16 *buffer_i, int j){
  /*load first half of pcm_next, second part of pcm_now.*/
  /*second part of pcm_next retains previous data */
  int i, k;
  int half_window = self->window_size / 2;
  ClassifyChannel *c = &self->channels[j];
  GST_LOG("buffer offset %d, channel %d",
      self->read_offset, j);

  for(i = 0, k = j; i < half_window; i++, k += self->n_channels){
    c->pcm_next[i] = buffer_i[k];
    c->pcm_now[half_window + i] = buffer_i[k];
  }
  /*get the features -- after which pcm_now is finished with. */
  pcm_to_features(self->mfcc_factory, c, self->mfccs, self->delta_features,
      self->intensity_feature);
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
train_channel(GstClassify *self, ClassifyChannel *c, int *win_count){
  RecurNN *net = c->net;
  float *answer = rnn_opinion(net, c->features, net->presynaptic_noise);
  float *error = net->bptt->o_error;
  float wrongness = 0;
  for (int i = 0; i < self->n_groups; i++){
    ClassifyClassGroup *g = &self->class_groups[i];
    int o = g->offset;
    float *group_error = error + o;
    float *group_answer = answer + o;
    int target = c->group_target[i];
    if (target >= 0 && target < g->n_classes &&
        self->window_no >= self->ignored_windows){
      int winner = softmax_best_guess(group_error, group_answer, g->n_classes);
      c->group_winner[i] = winner;
      *win_count += winner == target;
      group_error[target] += 1.0f;
      wrongness += group_error[target];
    }
    else {
      for (int j = 0; j < g->n_classes; j++){
        group_error[j] = 0;
      }
    }
  }
  if (self->error_weight){
    for (int i = 0; i < net->output_size; i++){
      error[i] *= self->error_weight[i];
    }
  }
  rnn_bptt_calc_deltas(net, 1);
  rnn_bptt_advance(net);
  return wrongness;
}

static inline s16 *
prepare_next_chunk(GstClassify *self){
  /*change the target classes of channels where necessary*/
  int half_window = self->window_size / 2;
  int chunk_size = half_window * self->n_channels;
  int len = self->write_offset - self->read_offset;
  if (len < 0){
    len += self->queue_size;
  }
  GST_LOG("start %d end %d len %d chunk %d queue_size %d", self->read_offset,
      self->write_offset, len, chunk_size, self->queue_size);
  if (len < chunk_size){
    GST_LOG("returning NULL");
    return NULL;
  }
  s16 *buffer = self->audio_queue + self->read_offset;
  self->read_offset += chunk_size;
  if (self->read_offset >= self->queue_size){
    self->read_offset -= self->queue_size;
    if (self->read_offset){
      /*the returned buffer will run beyond the end of the queue, where the
        memory is allocated but not initialised. So we copy the samples in
        from the beginning.*/
      memcpy(self->audio_queue + self->queue_size, self->audio_queue,
          self->read_offset * sizeof(s16));
    }
  }

  while(self->class_events_index < self->n_class_events){
    ClassifyClassEvent *ev = &self->class_events[self->class_events_index];
    if (ev->window_no > self->window_no){
      break;
    }
    ClassifyChannel *c = &self->channels[ev->channel];
    c->group_target[ev->class_group] = ev->target;
    GST_DEBUG("event %d/%d: channel %d.%d target -> %d at window  %d (%d)",
        self->class_events_index, self->n_class_events,
        ev->channel, ev->class_group, ev->target,
        self->window_no, ev->window_no);
    self->class_events_index++;
  }

  self->window_no++;
  GST_LOG("returning %p", buffer);

  return buffer;
}

static inline void
maybe_learn(GstClassify *self){
  int j;
  s16 *buffer;
  RecurNN *net = self->net;
  GST_LOG("maybe learn; offset %d",
      self->read_offset);

  while ((buffer = prepare_next_chunk(self))){
    float err_sum = 0.0f;
    int winners = 0;
    rnn_bptt_clear_deltas(net);
    GST_LOG("buffer offset %ld, %d", buffer - self->audio_queue,
        self->read_offset);
    for (j = 0; j < self->n_channels; j++){
      ClassifyChannel *c = prepare_channel_features(self, buffer, j);
      err_sum += train_channel(self, c, &winners);
    }

    /*XXX periodic_pgm_dump and image string should be gst properties */
    if (PERIODIC_PGM_DUMP && net->generation % PERIODIC_PGM_DUMP == 0){
      rnn_multi_pgm_dump(net, PERIODIC_PGM_DUMPEES, self->basename);
    }
    float momentum = rnn_calculate_momentum_soft_start(net->generation,
        net->bptt->momentum, self->momentum_soft_start);

    rnn_apply_learning(net, self->learning_style, momentum);
    rnn_condition_net(net);
    possibly_save_net(net, self->net_filename);
    rnn_log_net(net);
    if (self->error_image){
      temporal_ppm_row_from_source(self->error_image);
    }
    rnn_log_float(net, "momentum", momentum);
    rnn_log_float(net, "error", err_sum / self->n_channels);
    rnn_log_float(net, "correct", winners * 1.0f / self->n_channels);
  }
}

static inline void
emit_opinions(GstClassify *self, GstClockTime pts){
  int i, j;
  s16 *buffer;
  const float window_no_to_secs = (float)self->window_size * 0.5f / CLASSIFY_RATE;
  for (buffer = prepare_next_chunk(self); buffer;
       buffer = prepare_next_chunk(self)){
    float err_sum = 0.0f;
    int n_err = 0;
    float now = (float)self->window_no * window_no_to_secs - self->lag;
    for (j = 0; j < self->n_channels; j++){
      ClassifyChannel *c = prepare_channel_features(self, buffer, j);
      RecurNN *net = c->net;
      float *error = net->bptt->o_error;
      float *answer = rnn_opinion(net, c->features, 0);
      for (i = 0; i < self->n_groups; i++){
        ClassifyClassGroup *g = &self->class_groups[i];
        int o = g->offset;
        float *group_error = error + o;
        float *group_answer = answer + o;
        int winner = softmax_best_guess(group_error, group_answer, g->n_classes);
        c->group_winner[i] = winner;
        int target = c->group_target[i];
        if (target >= 0 && target < g->n_classes){
          err_sum += group_error[target];
          n_err++;
        }
      }
    }
    if (self->window_no >= self->ignored_windows){
      send_message(self, n_err ? err_sum / (float)n_err : 0, now, pts);
    }
  }
}


static GstFlowReturn
gst_classify_transform_ip (GstBaseTransform * base, GstBuffer *buf)
{
  GstClassify *self = GST_CLASSIFY(base);
  GstFlowReturn ret = GST_FLOW_OK;
  queue_audio_segment(buf, self->audio_queue, self->queue_size,
      &self->read_offset, &self->write_offset);
  if (self->training){
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
