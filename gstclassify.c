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
  PROP_MOMENTUM,
  PROP_MOMENTUM_STYLE,
  PROP_MOMENTUM_SOFT_START,
  PROP_MFCCS,
  PROP_SAVE_NET,
  PROP_PGM_DUMP,
  PROP_LOG_FILE,
  PROP_TRAINING,
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
  PROP_BOTTOM_LAYER,
  PROP_RANDOM_ALIGNMENT,

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
#define DEFAULT_PROP_MOMENTUM 0.95f
#define DEFAULT_PROP_MOMENTUM_SOFT_START 0.0f
#define DEFAULT_PROP_MOMENTUM_STYLE 1
#define DEFAULT_MIN_FREQUENCY 100
#define DEFAULT_MAX_FREQUENCY (CLASSIFY_RATE * 0.499)
#define MINIMUM_AUDIO_FREQUENCY 0
#define MAXIMUM_AUDIO_FREQUENCY (CLASSIFY_RATE * 0.5)

#define DEFAULT_PROP_CLASSES "tf"
#define DEFAULT_PROP_BPTT_DEPTH 30
#define DEFAULT_PROP_FORGET 0
#define DEFAULT_PROP_BOTTOM_LAYER 0
#define DEFAULT_PROP_WEIGHT_SPARSITY 1
#define DEFAULT_PROP_RANDOM_ALIGNMENT 1
#define DEFAULT_WINDOW_SIZE 256
#define DEFAULT_HIDDEN_SIZE 199
#define DEFAULT_LEARN_RATE 0.0001
#define DEFAULT_TOP_LEARN_RATE_SCALE 1.0f
#define DEFAULT_BOTTOM_LEARN_RATE_SCALE 1.0f
#define DEFAULT_PROP_DROPOUT 0.0f
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
#define PROP_BOTTOM_LAYER_MIN 0
#define PROP_BOTTOM_LAYER_MAX 1000000

#define LEARN_RATE_SCALE_MAX 1e9f
#define LEARN_RATE_SCALE_MIN 0

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
init_channel(ClassifyChannel *c, RecurNN *net,
    int window_size, int id, int n_groups)
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
  free(c->features);
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
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_PGM_DUMP,
      g_param_spec_string("pgm-dump", "pgm-dump",
          "Dump weight images (space separated \"ih* hh* ho* bi*\", *one of \"wdma\")",
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
      g_param_spec_string("classes", "classes",
          "Identify classes (one letter per class, groups separated by '|')",
          DEFAULT_PROP_CLASSES,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

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

  g_object_class_install_property (gobject_class, PROP_MIN_FREQUENCY,
      g_param_spec_float("min-frequency", "min-frequency",
          "Lowest audio frequency to analyse",
          MINIMUM_AUDIO_FREQUENCY, MAXIMUM_AUDIO_FREQUENCY,
          DEFAULT_MIN_FREQUENCY,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MAX_FREQUENCY,
      g_param_spec_float("max-frequency", "max-frequency",
          "Highest audio frequency to analyse",
          MINIMUM_AUDIO_FREQUENCY, MAXIMUM_AUDIO_FREQUENCY,
          DEFAULT_MAX_FREQUENCY,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

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
  self->audio_queue = NULL;
  self->net_filename = NULL;
  self->training = DEFAULT_PROP_TRAINING;
  self->pending_properties = calloc(PROP_LAST, sizeof(GValue));
  self->class_events = NULL;
  self->class_events_index = 0;
  self->n_class_events = 0;
  self->window_size = DEFAULT_WINDOW_SIZE;
  self->momentum_soft_start = DEFAULT_PROP_MOMENTUM_SOFT_START;
  self->dropout = DEFAULT_PROP_DROPOUT;
  self->basename = strdup(DEFAULT_BASENAME);
  self->error_weight = NULL;
  GST_INFO("gst classify init\n");
}


static void
reset_net_filename(GstClassify *self, int hidden_size, int bottom_layer,
    int top_layer_size){
  char s[200];
  int n_features = self->mfccs ? self->mfccs : CLASSIFY_N_FFT_BINS;
  if (bottom_layer > 0){
    snprintf(s, sizeof(s), "%s-i%d-b%d-h%d-o%d-b%d-%dHz-w%d.net",
        self->basename, n_features, bottom_layer, hidden_size, top_layer_size,
        CLASSIFY_BIAS, CLASSIFY_RATE, self->window_size);
  }
  else {
    snprintf(s, sizeof(s), "%s-i%d-h%d-o%d-b%d-%dHz-w%d.net",
        self->basename, n_features, hidden_size, top_layer_size,
        CLASSIFY_BIAS, CLASSIFY_RATE, self->window_size);
  }
  if (self->net_filename){
    free(self->net_filename);
  }
  self->net_filename = strdup(s);
}


static int parse_classes_string(GstClassify *self, const char *orig)
{
  char *str = strdup(orig);
  char *s;
  int i;
  int n_groups = 1;
  for (s = str; *s; s++){
    if (*s == '|'){
      *s = 0;
      n_groups++;
    }
  }
  self->class_groups = realloc_or_die(self->class_groups,
      (n_groups + 2) * sizeof(ClassifyClassGroup));
  s = str;
  int offset = 0;
  for (i = 0; i < n_groups; i++){
    ClassifyClassGroup *group = &self->class_groups[i];
    group->classes = s;
    group->n_classes = 1;
    group->offset = offset;
    for (; *s; s++){
      group->n_classes++;
      offset++;
    }
    s++;
  }
  self->n_groups = n_groups;
  return offset;
}


static RecurNN *
load_or_create_net(GstClassify *self){
  int hidden_size = get_gvalue_int(PENDING_PROP(self, PROP_HIDDEN_SIZE),
      DEFAULT_HIDDEN_SIZE);
  int bottom_layer_size = get_gvalue_int(PENDING_PROP(self, PROP_BOTTOM_LAYER), 0);

  int top_layer_size = parse_classes_string(self,
      get_gvalue_string(PENDING_PROP(self, PROP_CLASSES)));

  if (self->net_filename == NULL){
    reset_net_filename(self, hidden_size, bottom_layer_size, top_layer_size);
  }
  RecurNN *net = TRY_RELOAD ? rnn_load_net(self->net_filename) : NULL;
  if (net){
    if (net->output_size != top_layer_size){
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

#define GET_FLOAT(id, _default) get_gvalue_float(PENDING_PROP(self, id), _default)
#define GET_INT(id, _default) get_gvalue_int(PENDING_PROP(self, id), _default)

    int weight_sparsity = GET_INT(PROP_WEIGHT_SPARSITY, DEFAULT_PROP_WEIGHT_SPARSITY);
    int bptt_depth = GET_INT(PROP_BPTT_DEPTH, DEFAULT_PROP_BPTT_DEPTH);

    float momentum = GET_FLOAT(PROP_MOMENTUM, DEFAULT_PROP_MOMENTUM);
    float learn_rate = GET_FLOAT(PROP_LEARN_RATE, DEFAULT_LEARN_RATE);

    float bottom_learn_rate_scale = GET_FLOAT(PROP_BOTTOM_LEARN_RATE_SCALE,
        DEFAULT_BOTTOM_LEARN_RATE_SCALE);
    float top_learn_rate_scale = GET_FLOAT(PROP_TOP_LEARN_RATE_SCALE,
        DEFAULT_TOP_LEARN_RATE_SCALE);

    float fan_in_sum = GET_FLOAT(PROP_WEIGHT_FAN_IN_SUM,
        DEFAULT_PROP_WEIGHT_FAN_IN_SUM);

    float fan_in_kurtosis = GET_FLOAT(PROP_WEIGHT_FAN_IN_KURTOSIS,
        DEFAULT_PROP_WEIGHT_FAN_IN_KURTOSIS);

    u64 rng_seed = get_gvalue_u64(PENDING_PROP(self, PROP_RNG_SEED), DEFAULT_RNG_SEED);
    GST_DEBUG("rng seed %lu", rng_seed);

#undef GET_FLOAT
#undef GET_INT

    net = rnn_new_with_bottom_layer(n_features, bottom_layer_size, hidden_size,
        top_layer_size, flags, rng_seed,
        NULL, bptt_depth, learn_rate, momentum, 0);
    if (fan_in_sum){
      rnn_randomise_weights_fan_in(net, fan_in_sum, fan_in_kurtosis, 0.1f, 0);
    }
    else {
      rnn_randomise_weights(net, RNN_INITIAL_WEIGHT_VARIANCE_FACTOR / net->h_size,
          weight_sparsity, 0.5);
    }
    net->bptt->ho_scale = top_learn_rate_scale;
    if (net->bottom_layer){
      net->bottom_layer->learn_rate_scale = bottom_learn_rate_scale;
    }
    if (PERIODIC_PGM_DUMP){
      rnn_multi_pgm_dump(net, "how ihw iha biw bid");
    }
  }
  if (net->bottom_layer){
    self->error_image = temporal_ppm_alloc(net->bottom_layer->o_size, 300, "bottom_error",
        0, PGM_DUMP_COLOUR, &net->bottom_layer->o_error);
  }
  return net;
}


static gboolean
gst_classify_setup(GstAudioFilter *base, const GstAudioInfo *info){
  GST_INFO("gst_classify_setup\n");
  GstClassify *self = GST_CLASSIFY(base);
  if (self->mfcc_factory == NULL){
    float min_freq = get_gvalue_float(PENDING_PROP(self, PROP_MIN_FREQUENCY),
      DEFAULT_MIN_FREQUENCY);
    float max_freq = get_gvalue_float(PENDING_PROP(self, PROP_MAX_FREQUENCY),
      DEFAULT_MIN_FREQUENCY);

    self->mfcc_factory = recur_audio_binner_new(self->window_size,
        RECUR_WINDOW_HANN,
        CLASSIFY_N_FFT_BINS,
        min_freq, max_freq,
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
    if (self->audio_queue){
      free(self->audio_queue);
    }
    self->n_channels = info->channels;
    self->queue_size = info->channels * self->window_size * CLASSIFY_QUEUE_FACTOR;
    int alloc_size = self->queue_size + info->channels * self->window_size / 2;
    self->audio_queue = malloc_aligned_or_die(alloc_size * sizeof(s16));
    if (self->channels){
      free(self->channels);
    }
    self->channels = malloc_aligned_or_die(self->n_channels * sizeof(ClassifyChannel));
    if (self->subnets){
      free(self->subnets);
    }
    self->subnets = rnn_new_training_set(self->net, self->n_channels);
    /*XXXXX init_channel allocates memory*/
    for (int i = 0; i < self->n_channels; i++){
      init_channel(&self->channels[i], self->subnets[i],
          self->window_size, i, self->n_groups);
    }
  }
  maybe_start_logging(self);
  maybe_parse_target_string(self);

  GstStructure *s = gst_structure_new_empty("classify-setup");
  GstMessage *msg = gst_message_new_element(GST_OBJECT(self), s);
  gst_element_post_message(GST_ELEMENT(self), msg);
  if (self->random_alignment){
    self->write_offset = 0;
    int offset = rand_small_int(&self->net->rng, self->window_size) - self->window_size / 2;
    self->read_offset = offset * self->n_channels;
    if (offset < 0){
      self->read_offset += self->queue_size;
      STDERR_DEBUG("random offset is %d, read_offset %d queue_size %d", offset,
          self->read_offset, self->queue_size);
      memset(self->audio_queue + self->read_offset, 0,
          (self->queue_size - self->read_offset) * sizeof(s16));
    }

    STDERR_DEBUG("random offset is %d (%d * %d)",
        self->read_offset, offset, self->n_channels);
  }
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

  For example, if the classes had been set to "Mm|Kk|Wx", this would be a
  valid target string:

  "c0t2.3:mkW c0t4.1:m-x c11t123:=K= c1t0:Mkx"

  All channels and categories start off in an untargetted state, as if they
  had been passed the '-' token.
 */
//XXX look at test_backprop capital learning for multi-softmax.
static inline int
parse_complex_target_string(GstClassify *self, const char *str){
  char *e;
  const char *s;
  int i;
  float time_to_window_no = CLASSIFY_RATE * 2.0f / self->window_size;
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

    window_no = time * time_to_window_no + 0.5;
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
        (double)ev->window_no * self->window_size / (2.0f * CLASSIFY_RATE), time);
#undef ERROR_IF
  }
  qsort(self->class_events, n_events, sizeof(ClassifyClassEvent), cmp_class_event);
  self->n_class_events = n_events;
  self->class_events_index = 0;
#if 0
  for (i = 0; i < n_events; i++){
    ev = &self->class_events[i];
    fprintf(stderr, "c%dt%.2f:%c ",
        ev->channel, ev->window_no / time_to_window_no,
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
  char *orig, *e, *s;
  int i;
  if (self->channels == NULL){
    GST_DEBUG("not parsing target string because channels is NULL");
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
  const char *s = get_gvalue_string(PENDING_PROP(self, PROP_LOG_FILE));
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
    case PROP_BOTTOM_LEARN_RATE_SCALE:
      if (net->bottom_layer){
        SET_FLOAT(net->bottom_layer->learn_rate_scale);
      }
      break;
    }
  }
  else {
    set_gvalue(PENDING_PROP(self, prop_id), value);
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

    case PROP_RANDOM_ALIGNMENT:
      self->random_alignment = g_value_get_boolean(value);

    case PROP_TRAINING:
      self->training = g_value_get_boolean(value);
      break;

    case PROP_DROPOUT:
      self->dropout = g_value_get_float(value);
      break;

    case PROP_MOMENTUM_SOFT_START:
      self->momentum_soft_start = g_value_get_float(value);
      break;

    case PROP_MOMENTUM_STYLE:
      self->momentum_style = g_value_get_int(value);
      break;

    case PROP_TOP_LEARN_RATE_SCALE:
    case PROP_BOTTOM_LEARN_RATE_SCALE:
    case PROP_LEARN_RATE:
    case PROP_MOMENTUM:
      maybe_set_net_scalar(self, prop_id, value);
      break;
    /*properties that only need to be stored until net creation, and can't
      be changed afterwards go here.
    */
    case PROP_MIN_FREQUENCY:
    case PROP_MAX_FREQUENCY:
    case PROP_CLASSES:
    case PROP_BOTTOM_LAYER:
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

#undef SET_INT_IF_NOT_TOO_LATE

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
    }
  }
}

void
maybe_get_net_scalar(GstClassify *self, guint prop_id, GValue *dest)
{
  RecurNN *net = self->net;
  if (net){
    switch (prop_id){
    case PROP_LEARN_RATE:
      g_value_set_float(dest, net->bptt->learn_rate);
      break;
    case PROP_BPTT_DEPTH:
      g_value_set_int(dest, net->bptt->depth);
      break;
    case PROP_MOMENTUM:
      g_value_set_float(dest, net->bptt->momentum);
      break;
    case PROP_HIDDEN_SIZE:
      g_value_set_int(dest, net->hidden_size);
      break;
    case PROP_BOTTOM_LAYER:
      {
        int x = (net->bottom_layer) ? net->bottom_layer->output_size : 0;
        g_value_set_int(dest, x);
      }
      break;
    case PROP_TOP_LEARN_RATE_SCALE:
      g_value_set_float(dest, net->bptt->ho_scale);
      break;
    case PROP_BOTTOM_LEARN_RATE_SCALE:
      if (net->bottom_layer){
        g_value_set_float(dest, net->bottom_layer->learn_rate_scale);
      }
      else {
        g_value_set_float(dest, 0);/* there is no right answer */
      }
      break;
    case PROP_LAWN_MOWER:
      g_value_set_boolean(dest, !! (self->net->flags & RNN_COND_USE_LAWN_MOWER));
      break;
    }
  }
  else {
    const GValue *src = PENDING_PROP(self, prop_id);
    if (G_IS_VALUE(src)){
      g_value_copy(src, dest);
    }
    else {
      /*now what?*/
    }
  }
}

static void
gst_classify_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstClassify *self = GST_CLASSIFY (object);
  switch (prop_id) {
  case PROP_LEARN_RATE:
  case PROP_TOP_LEARN_RATE_SCALE:
  case PROP_MOMENTUM:
  case PROP_BOTTOM_LEARN_RATE_SCALE:
  case PROP_BPTT_DEPTH:
  case PROP_HIDDEN_SIZE:
  case PROP_BOTTOM_LAYER:
  case PROP_LAWN_MOWER:
      maybe_get_net_scalar(self, prop_id, value);
      break;

  case PROP_TRAINING:
    g_value_set_boolean(value, self->training);

  case PROP_MFCCS:
    g_value_set_int(value, self->mfccs);
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

  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    break;
  }
}

/*XXX shared with parrot and possibly others */
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
    char key[60];
    char id[40];
    ClassifyChannel *c = &self->channels[i];
    RecurNN *net = c->net;
    for (int j = 0; j < self->n_groups; j++){
      ClassifyClassGroup *g = &self->class_groups[j];
      int target = c->group_target[j];
      int winner = c->group_winner[j];
      char t_char = g->classes[target];
      char w_char = g->classes[winner];
      //STDERR_DEBUG("classes %s target %d winner %d t_char %d w_char %d",
      //    g->classes, target, winner, t_char, w_char);
      snprintf(id, sizeof(id), "channel %d, group %d", i, j);
      for (int k = 0; k < g->n_classes; k++){
        int t = g->classes[k];
        float error = net->bptt->o_error[g->offset + k];
        snprintf(key, sizeof(key), "%s %c", id, t);
        gst_structure_set(s, key, G_TYPE_FLOAT, -error, NULL);
      }
      if (target >= 0 && target < g->n_classes){
        snprintf(key, sizeof(key), "%s correct", id);
        gst_structure_set(s, key, G_TYPE_INT, winner == target, NULL);
        snprintf(key, sizeof(key), "%s target", id);
        gst_structure_set(s, key, G_TYPE_CHAR, t_char, NULL);
      }
      snprintf(key, sizeof(key), "%s winner", id);
      gst_structure_set(s, key, G_TYPE_CHAR, w_char, NULL);
    }
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
  GST_LOG("buffer offset %d, channel %d",
      self->read_offset, j);

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
train_channel(GstClassify *self, ClassifyChannel *c, int *win_count){
  RecurNN *net = c->net;
  float *answer = rnn_opinion(net, c->features, self->dropout);
  float *error = net->bptt->o_error;
  float wrongness = 0;
  for (int i = 0; i < self->n_groups; i++){
    ClassifyClassGroup *g = &self->class_groups[i];
    int o = g->offset;
    float *group_error = error + o;
    float *group_answer = answer + o;
    int target = c->group_target[i];
    int winner = softmax_best_guess(group_error, group_answer, g->n_classes);
    c->group_winner[i] = winner;
    *win_count += winner == target;
    group_error[target] += 1.0f;
    wrongness += group_error[target];
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
      rnn_multi_pgm_dump(net, "how ihw iha biw bid");
    }
    float momentum = rnn_calculate_momentum_soft_start(net->generation,
        net->bptt->momentum, self->momentum_soft_start);

    rnn_apply_learning(net, self->momentum_style, momentum);
    rnn_condition_net(net);
    possibly_save_net(net, self->net_filename);
    rnn_log_net(net);
    if (self->error_image){
      temporal_ppm_row_from_source(self->error_image);
    }

    rnn_log_float(net, "error", err_sum / self->n_channels);
    rnn_log_float(net, "correct", winners * 1.0 / self->n_channels);
  }
}

static inline void
emit_opinions(GstClassify *self, GstClockTime pts){
  int i, j;
  s16 *buffer;
  for (buffer = prepare_next_chunk(self); buffer;
       buffer = prepare_next_chunk(self)){
    float err_sum = 0.0f;
    int n_err = 0;
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
    send_message(self, n_err ? err_sum / n_err : 0, pts);
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
