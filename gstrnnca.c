/* GStreamer
 * Copyright (C) <2013> Douglas Bagnall <douglas@halo.gen.nz>
 *
 */

#include "gstrnnca.h"
#include "rescale.h"
#include "blit-helpers.h"
#include "recur-common.h"
#include <gst/video/gstvideofilter.h>
#include <gst/video/video.h>

#include <string.h>
#include <math.h>


GST_DEBUG_CATEGORY_STATIC (rnnca_debug);
#define GST_CAT_DEFAULT rnnca_debug

enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_LEARN_RATE,
  PROP_HIDDEN_SIZE,
  PROP_SAVE_NET,
  PROP_PGM_DUMP,
  PROP_LOG_FILE,
  PROP_TRAINING,
  PROP_PLAYING,
  PROP_EDGES,
  PROP_MOMENTUM_SOFT_START,
  PROP_MOMENTUM,
};


#define DEFAULT_PROP_PGM_DUMP ""
#define DEFAULT_PROP_LOG_FILE ""
#define DEFAULT_PROP_SAVE_NET NULL
#define DEFAULT_PROP_PLAYING 1
#define DEFAULT_PROP_TRAINING 1
#define DEFAULT_PROP_EDGES 0
#define DEFAULT_HIDDEN_SIZE (52 - RNNCA_BIAS)
#define DEFAULT_LEARN_RATE 3e-4
#define MIN_HIDDEN_SIZE 1
#define MAX_HIDDEN_SIZE 1000000
#define LEARN_RATE_MIN 0.0
#define LEARN_RATE_MAX 1.0
#define DEFAULT_PROP_MOMENTUM 0.95f
#define DEFAULT_PROP_MOMENTUM_SOFT_START 0.0f
#define MOMENTUM_MIN 0.0
#define MOMENTUM_MAX 1.0
#define MOMENTUM_SOFT_START_MAX 1e9
#define MOMENTUM_SOFT_START_MIN 0


/* static_functions */
/* plugin_init    - registers plugin (once)
   XXX_base_init  - for the gobject class (once, obsolete)
   XXX_class_init - for global state (once)
   XXX_init       - for each plugin instance
*/
static void gst_rnnca_class_init(GstRnncaClass *g_class);
static void gst_rnnca_init(GstRnnca *self);
static void gst_rnnca_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void gst_rnnca_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static GstFlowReturn gst_rnnca_transform_frame_ip(GstVideoFilter *base, GstVideoFrame *buf);

static void maybe_set_learn_rate(GstRnnca *self);


static gboolean set_info (GstVideoFilter *filter,
    GstCaps *incaps, GstVideoInfo *in_info,
    GstCaps *outcaps, GstVideoInfo *out_info);


#define VIDEO_FORMATS " { I420 } "

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE (VIDEO_FORMATS))
    );

static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE (VIDEO_FORMATS))
    );

#define gst_rnnca_parent_class parent_class
G_DEFINE_TYPE (GstRnnca, gst_rnnca, GST_TYPE_VIDEO_FILTER)

/* Clean up */
static void
gst_rnnca_finalize (GObject * obj){
  GST_DEBUG("in gst_rnnca_finalize!\n");
  GstRnnca *self = GST_RNNCA(obj);
  if (self->frame_prev && self->frame_now){
    free(self->frame_prev->Y);
    free(self->frame_now->Y);
    free(self->frame_now);
    free(self->frame_prev);
  }
  if (self->training_map){
    free(self->training_map);
  }
}

static void
gst_rnnca_class_init (GstRnncaClass * g_class)
{
  GST_DEBUG_CATEGORY_INIT (rnnca_debug, "rnnca", RECUR_LOG_COLOUR,
      "rnnca video");

  //GstBaseTransformClass *trans_class = GST_BASE_TRANSFORM_CLASS (g_class);
  GstElementClass *gstelement_class = (GstElementClass *) g_class;

  GObjectClass *gobject_class = G_OBJECT_CLASS (g_class);
  GstVideoFilterClass *vf_class = GST_VIDEO_FILTER_CLASS (g_class);

  gobject_class->set_property = gst_rnnca_set_property;
  gobject_class->get_property = gst_rnnca_get_property;
  gobject_class->finalize = GST_DEBUG_FUNCPTR (gst_rnnca_finalize);

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));

  gst_element_class_set_static_metadata (gstelement_class,
      "RNN Cellular automata video element",
      "Filter/Video",
      "Mangles video",
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
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));
  /*XXX make these readwrite*/
  g_object_class_install_property (gobject_class, PROP_PLAYING,
      g_param_spec_boolean("playing", "playing",
          "Construct imaginary video",
          DEFAULT_PROP_PLAYING,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_TRAINING,
      g_param_spec_boolean("training", "training",
          "Learn from incoming video",
          DEFAULT_PROP_TRAINING,
          G_PARAM_WRITABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_EDGES,
      g_param_spec_boolean("edges", "edges",
          "Play on edged rectangle, not torus",
          DEFAULT_PROP_EDGES,
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

  vf_class->transform_frame_ip = GST_DEBUG_FUNCPTR (gst_rnnca_transform_frame_ip);
  vf_class->set_info = GST_DEBUG_FUNCPTR (set_info);
  GST_INFO("gst class init\n");
}

static void
gst_rnnca_init (GstRnnca * self)
{
  self->net = NULL;
  self->frame_prev = NULL;
  self->frame_now = NULL;
  self->play_frame = NULL;
  self->constructors = NULL;
  self->trainers = NULL;
  self->training_map = NULL;
  self->training = 1;
  self->playing = 1;
  self->edges = DEFAULT_PROP_EDGES;
  self->hidden_size = DEFAULT_HIDDEN_SIZE;
  self->pending_learn_rate = 0;
  self->momentum_soft_start = DEFAULT_PROP_MOMENTUM_SOFT_START;
  self->momentum = DEFAULT_PROP_MOMENTUM;
  GST_INFO("gst rnnca init\n");
}

static void
reset_net_filename(GstRnnca *self){
  char s[200];
  snprintf(s, sizeof(s), "rnnca-i%d-h%d-o%d-b%d-yuv%d-y%d-x%d.net",
      RNNCA_N_FEATURES, self->hidden_size, 3,
      RNNCA_BIAS, RNNCA_YUV_LEN, RNNCA_Y_ONLY_LEN, 2);
  if (self->net_filename){
    free(self->net_filename);
  }
  self->net_filename = strdup(s);
}

static int
compare_trainers(const void *a, const void *b){
  const RnncaTrainer *at = (RnncaTrainer *)a;
  const RnncaTrainer *bt = (RnncaTrainer *)b;
  return (at->y * RNNCA_WIDTH + at->x) - (bt->y * RNNCA_WIDTH + bt->x);
}

const int TRAINER_MARGIN = 2;

static int
randomly_place_trainer(RnncaTrainer *t, rand_ctx *rng, u8 *mask){
  int i;
  const int w = RNNCA_WIDTH;
  const int h = RNNCA_HEIGHT;
  for (i = 0; i < 20; i++){
    int x = TRAINER_MARGIN + rand_small_int(rng, w - 2 * TRAINER_MARGIN);
    int y = TRAINER_MARGIN + rand_small_int(rng, h - 2 * TRAINER_MARGIN);
    if (! mask[y * w + x]){
      mask[y * w + x] = 255;
      t->x = x;
      t->y = y;
      return 0;
    }
  }
  GST_WARNING("could not place trainer after %d goes. this should not be!", i);
  pgm_dump(mask, w, h, IMAGE_DIR "mask-broken.pgm");
  return 1;
}

static void
construct_trainers(GstRnnca *self, int n_requested)
{
  int i, j;
  RecurNN *net = self->net;
  u32 flags = self->net->flags & ~RNN_NET_FLAG_OWN_WEIGHTS;
  const int w = RNNCA_WIDTH;
  const int h = RNNCA_HEIGHT;
  u8* mask = zalloc_aligned_or_die(w * h);
  self->training_map = mask;
  self->trainers = malloc_aligned_or_die(n_requested * sizeof(RnncaTrainer));
  self->train_nets = malloc_aligned_or_die(n_requested * sizeof(RecurNN*));
  for (j = 0, i = 0; i < n_requested * 2; i++) {
    RnncaTrainer *t = &self->trainers[j];
    if(!randomly_place_trainer(t, &net->rng, mask)){
      RecurNN *clone = rnn_clone(net, flags, RECUR_RNG_SUBSEED, NULL);
      self->trainers[j].net = clone;
      self->train_nets[j] = clone;
      GST_LOG("net %d is %p", j, clone);
      j++;
      if (j == n_requested){
        goto done;
      }
    }
  }
  GST_ERROR("Could only fit %d out of %d desired training nets", j, n_requested);
 done:
  qsort(self->trainers, j, sizeof(RnncaTrainer), compare_trainers);
  self->n_trainers = j;
  pgm_dump(mask, w, h, IMAGE_DIR "mask.pgm");
}



static RecurNN *
load_or_create_net(GstRnnca *self){
  reset_net_filename(self);
  RecurNN *net = TRY_RELOAD ? rnn_load_net(self->net_filename) : NULL;
  if (net == NULL){
    net = rnn_new(RNNCA_N_FEATURES, self->hidden_size, 3,
        RNNCA_RNN_FLAGS, RNNCA_RNG_SEED,
        NULL, RNNCA_BPTT_DEPTH, DEFAULT_LEARN_RATE, self->momentum, MOMENTUM_WEIGHT,
        RNNCA_BATCH_SIZE, 0);
    rnn_randomise_weights(net, 0.2f, 1);
    //net->bptt->ho_scale = 0.25;
  }
  else {
    rnn_set_log_file(net, NULL, 0);
  }
  maybe_set_learn_rate(self);
  return net;
}

static void
maybe_start_logging(GstRnnca *self){
  if (self->pending_logfile && self->trainers){
    if (self->pending_logfile[0] == 0){
      rnn_set_log_file(self->trainers[0].net, NULL, 0);
    }
    else {
      rnn_set_log_file(self->trainers[0].net, self->pending_logfile, 1);
    }
    free(self->pending_logfile);
    self->pending_logfile = NULL;
  }
}

static gboolean
set_info (GstVideoFilter *filter,
    GstCaps *incaps, GstVideoInfo *in_info,
    GstCaps *outcaps, GstVideoInfo *out_info)
{
  GstRnnca *self = GST_RNNCA (filter);
  int i;
  if (self->net == NULL){
    self->net = load_or_create_net(self);
  }
  if (self->constructors == NULL){
    int n = RNNCA_WIDTH * RNNCA_HEIGHT;
    self->constructors = malloc_aligned_or_die(n * sizeof(RecurNN *));
    for (i = 0; i < n; i++){
      u32 flags = self->net->flags & ~(RNN_NET_FLAG_OWN_WEIGHTS | RNN_NET_FLAG_OWN_BPTT);
      RecurNN *clone = rnn_clone(self->net, flags, RECUR_RNG_SUBSEED, NULL);
      self->constructors[i] = clone;
    }
  }

  if (self->frame_prev == NULL){
    self->frame_prev = malloc_aligned_or_die(sizeof(RnncaFrame));
    self->frame_now = malloc_aligned_or_die(sizeof(RnncaFrame));
    self->play_frame = malloc_aligned_or_die(sizeof(RnncaFrame));
    size_t size = RNNCA_WIDTH * RNNCA_HEIGHT;
    u8 *mem = zalloc_aligned_or_die(size * 3);
    self->frame_prev->Y  = mem;
    self->frame_prev->Cb = mem + size;
    self->frame_prev->Cr = mem + size * 2;
    mem = zalloc_aligned_or_die(size * 3);
    self->frame_now->Y  = mem;
    self->frame_now->Cb = mem + size;
    self->frame_now->Cr = mem + size * 2;
    mem = malloc_aligned_or_die(size * 3);
    randomise_mem(&self->net->rng, mem, size * 3);
    self->play_frame->Y  = mem;
    self->play_frame->Cb = mem + size;
    self->play_frame->Cr = mem + size * 2;
  }
  if (self->trainers == NULL){
    construct_trainers(self, RNNCA_N_TRAINERS);
  }
  maybe_start_logging(self);
  return TRUE;
}

static void
maybe_set_learn_rate(GstRnnca *self){
  float lr = self->pending_learn_rate;
  if (lr){
    if (self->net){
      self->net->bptt->learn_rate = lr;
      self->pending_learn_rate = 0;
      if (self->trainers){
        for (int i = 0; i < self->n_trainers; i++){
          self->trainers[i].net->bptt->learn_rate = lr;
        }
      }
    }
  }
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
gst_rnnca_set_property (GObject * object, guint prop_id, const GValue * value,
    GParamSpec * pspec)
{
  GstRnnca *self = GST_RNNCA (object);
  GST_DEBUG("gst_rnnca_set_property\n");
  if (value){
    switch (prop_id) {
    case PROP_LOG_FILE:
      /*defer setting the actual log file, in case the nets aren't ready yet*/
      if (self->pending_logfile){
        free(self->pending_logfile);
      }
      self->pending_logfile = g_value_dup_string(value);
      maybe_start_logging(self);
      break;

    case PROP_PLAYING:
      self->playing = g_value_get_boolean(value);
      break;

    case PROP_TRAINING:
      self->training = g_value_get_boolean(value);
      break;

    case PROP_EDGES:
      self->edges = g_value_get_boolean(value);
      break;

    case PROP_LEARN_RATE:
      self->pending_learn_rate = g_value_get_float(value);
      maybe_set_learn_rate(self);
      break;

    case PROP_MOMENTUM_SOFT_START:
      self->momentum_soft_start = g_value_get_float(value);
      break;

    case PROP_MOMENTUM:
      self->momentum = g_value_get_float(value);
      break;

    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
    }
  }
}

static void
gst_rnnca_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstRnnca *self = GST_RNNCA (object);

  switch (prop_id) {
  case PROP_LEARN_RATE:
    if (self->train_nets){
      g_value_set_float(value, self->train_nets[0]->bptt->learn_rate);
    }
    break;
  case PROP_MOMENTUM:
    g_value_set_float(value, self->momentum);
    break;
  case PROP_MOMENTUM_SOFT_START:
    g_value_set_float(value, self->momentum_soft_start);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
    break;
  }
}

static inline void
remember_frame(GstRnnca *self, GstVideoFrame *frame){
  int i;
  RnncaFrame *thumb = self->frame_prev;
  u8 *plane = thumb->Y;
  for (i = 0; i < 3; i++){
    /*convert first to working size */
    const u8 *src = GST_VIDEO_FRAME_COMP_DATA(frame, i);
    int sw = GST_VIDEO_FRAME_COMP_WIDTH(frame, i);
    int sh = GST_VIDEO_FRAME_COMP_HEIGHT(frame, i);
    int ss = GST_VIDEO_FRAME_COMP_STRIDE(frame, i);
    GST_DEBUG("thumb %p, plane %p (%x %x) sw %d sh %d",
        thumb, plane, plane[0], plane[1], sw, sh);
    recur_adaptive_downscale(src, sw, sh, ss,
        plane, RNNCA_WIDTH, RNNCA_HEIGHT, RNNCA_WIDTH);
    plane += RNNCA_WIDTH * RNNCA_HEIGHT;
  }
  self->frame_prev = self->frame_now;
  self->frame_now = thumb;
}

#define BYTE_TO_UNIT(x) ((x) * (1.0f / 255.0f))
#define BYTE_TO_BALANCED_UNIT(x) (((x) * (1.0f / 127.5f)) - 127.5f)
#define UNIT_TO_BYTE(x) ((x) * (255.9f))

static inline void
deal_with_edges(int *x, int *y, int edges)
{
  if (edges){
    *y = MAX(0, MIN(RNNCA_HEIGHT - 1, *y));
    *x = MAX(0, MIN(RNNCA_WIDTH - 1, *x));
  }
  else{
    if (*y < 0){
      *y += RNNCA_HEIGHT;
    }
    else if (*y >= RNNCA_HEIGHT){
      *y -= RNNCA_HEIGHT;
    }
    if (*x < 0){
      *x += RNNCA_WIDTH;
    }
    else if (*x >= RNNCA_WIDTH){
      *x -= RNNCA_WIDTH;
    }
  }
}


static inline void
fill_net_inputs(RecurNN *net, RnncaFrame *frame, int cx, int cy, int edges){
  int y, offset, iy, ix, j;
  int i = 0, x = 0;
  //GST_DEBUG("frame is %p, cx %d, cy %d", frame, cx, cy);
  for (j = 0; j < RNNCA_YUV_LEN; j+= 2){
    ix = RNNCA_YUV_OFFSETS[j];
    iy = RNNCA_YUV_OFFSETS[j + 1];
    y = cy + iy;
    x = cx + ix;
    deal_with_edges(&x, &y, edges);

    offset = y * RNNCA_WIDTH + x;
    net->real_inputs[i] = BYTE_TO_UNIT(frame->Y[offset]);
    net->real_inputs[i + 1] = BYTE_TO_UNIT(frame->Cb[offset]);
    net->real_inputs[i + 2] = BYTE_TO_UNIT(frame->Cr[offset]);
    i += 3;
  }
  for (j = 0; j < RNNCA_Y_ONLY_LEN; j+= 2){
    ix = RNNCA_YUV_OFFSETS[j];
    iy = RNNCA_YUV_OFFSETS[j + 1];
    y = cy + iy;
    x = cx + ix;
    deal_with_edges(&x, &y, edges);

    offset = y * RNNCA_WIDTH + x;
    net->real_inputs[i] = BYTE_TO_UNIT(frame->Y[offset]);
    i++;
  }
  net->real_inputs[i] = abs(cx - RNNCA_WIDTH) * 1.0 / RNNCA_WIDTH;
  net->real_inputs[i + 1] = abs(cy - RNNCA_HEIGHT) * 1.0 / RNNCA_HEIGHT;
}

static inline void
train_net(RnncaTrainer *t, RnncaFrame *prev,  RnncaFrame *now){
  int i, offset, plane_size;
  RecurNN *net = t->net;
  /*trainers are not on edges, so edge condition doesn't much matter */
  fill_net_inputs(net, prev, t->x, t->y, 1);

  float *il = net->real_inputs;
  GST_LOG("inputs %.2g %.2g %.2g  %.2g %.2g %.2g  %.2g %.2g %.2g   "
      "%.2g %.2g %.2g  %.2g %.2g %.2g  %.2g %.2g %.2g   "
      "%.2g %.2g %.2g  %.2g %.2g %.2g  %.2g %.2g %.2g   ",
      il[0], il[1], il[2], il[3], il[4], il[5], il[6], il[7], il[8],
      il[9], il[10], il[11], il[12], il[13], il[14], il[15], il[16], il[17],
      il[18], il[19], il[20], il[21], il[22], il[23], il[24], il[25], il[26]
  );

  float *answer = rnn_opinion(net, NULL);
  fast_sigmoid_array(answer, answer, 3);
  offset = t->y * RNNCA_WIDTH + t->x;
  GST_DEBUG("x %d, y %d, offset %d", t->x, t->y, offset);
  plane_size = RNNCA_WIDTH * RNNCA_HEIGHT;
  for (i = 0; i < 3; i++){
    GST_LOG("now %p prev %p Y %p/%p plane_size %d, offset %d",
        now, prev, now->Y, prev->Y, plane_size, offset);
    float target = BYTE_TO_UNIT(now->Y[offset + plane_size * i]);
    float a = answer[i];
    float slope = a * (1.0f - a);
    net->bptt->o_error[i] = slope * (target - a);
    GST_LOG("target %.2g a %.2g diff %.2g slope %.2g",
        target, a, target - a, slope);
  }
  bptt_calc_deltas(net);
}

static inline void
maybe_learn(GstRnnca *self){
  int i;
  RecurNN *net = self->train_nets[0];
  for (i = 0; i < self->n_trainers; i++){
    train_net(&self->trainers[i], self->frame_prev, self->frame_now);
  }
  bptt_consolidate_many_nets(self->train_nets, self->n_trainers, 0,
      self->momentum_soft_start);
  if (PERIODIC_PGM_DUMP && (net->generation & PERIODIC_PGM_DUMP) == 0){
    rnn_multi_pgm_dump(net, "how ihw");
  }
  rnn_log_net(net);
  self->net->generation = net->generation;
  rnn_condition_net(self->net);
  if (PERIODIC_SAVE_NET && (self->net->generation & PERIODIC_SAVE_NET) == 0){
    rnn_save_net(self->net, self->net_filename);
  }
  if (PERIODIC_SHUFFLE_TRAINERS &&
      (net->generation & PERIODIC_SHUFFLE_TRAINERS) == 0){
    i = rand_small_int(&net->rng, self->n_trainers);
    RnncaTrainer *t = &self->trainers[i];
    self->training_map[t->y * RNNCA_WIDTH + t->x] = 0;
    randomly_place_trainer(t, &net->rng, self->training_map);
#if PGM_DUMP_CHANGED_MASK
    char name[50 + sizeof(IMAGE_DIR)];
    snprintf(name, sizeof(name), "%smask-%u.pgm", IMAGE_DIR, net->generation);
    pgm_dump(self->training_map, RNNCA_WIDTH, RNNCA_HEIGHT, name);
    /*XXX maybe keep in sorted order ?*/
    GST_DEBUG("shifted trainer %d to %d,%d, map %s", i, t->x, t->y, name);
#endif
  }
}

static inline void
fill_frame(GstRnnca *self, GstVideoFrame *frame){
  int x, y, offset;
  if (PERIODIC_CHECK_STASIS &&
      (self->net->generation & PERIODIC_CHECK_STASIS) == 0){
    offset = 2 * RNNCA_WIDTH + rand_small_int(&self->net->rng, RNNCA_WIDTH);
    u8 Y = self->play_frame->Y[offset];
    u8 Cb = self->play_frame->Cb[offset];
    u8 Cr = self->play_frame->Cr[offset];
    const int m = 3;
    GST_DEBUG("row 2, found yuv %d %d %d", Y, Cb, Cr);
    for (y = 3; y < RNNCA_HEIGHT - 2; y++){
      x = rand_small_int(&self->net->rng, RNNCA_WIDTH);
      offset = y * RNNCA_WIDTH + x;
      u8 Y2 = self->play_frame->Y[offset];
      u8 Cb2 = self->play_frame->Cb[offset];
      u8 Cr2 = self->play_frame->Cr[offset];
      if (Y < Y2 - m || Y > Y2 + m ||
          Cb < Cb2 - m || Cb > Cb2 + m ||
          Cr < Cr2 - m || Cr > Cr2 + m){
        GST_DEBUG("row %i, found yuv %d %d %d", y, Y2, Cb2, Cr2);
        goto ok_its_not_flat;
      }
    }
    for (y = 1; y < RNNCA_HEIGHT; y++){
      x = rand_small_int(&self->net->rng, RNNCA_WIDTH);
      offset = y * RNNCA_WIDTH + x;
      self->play_frame->Y[offset] = rand_small_int(&self->net->rng, 256);
      self->play_frame->Cb[offset] = rand_small_int(&self->net->rng, 256);
      self->play_frame->Cr[offset] = rand_small_int(&self->net->rng, 256);
    }
  }
 ok_its_not_flat:
  for (y = 0; y < RNNCA_HEIGHT; y++){
    for (x = 0; x < RNNCA_WIDTH; x++){
      RecurNN *net = self->constructors[y * RNNCA_WIDTH + x];
      fill_net_inputs(net, self->play_frame, x, y, self->edges);
      float *answer = rnn_opinion(net, NULL);
      fast_sigmoid_array(answer, answer, 3);
      GST_LOG("answer gen %d, x %d y %d, %.2g %.2g %.2g",
          net->generation, x, y, answer[0], answer[1], answer[2]);
    }
  }
  for (y = 0; y < RNNCA_HEIGHT; y++){
    for (x = 0; x < RNNCA_WIDTH; x++){
      offset = y * RNNCA_WIDTH + x;
      float *yuv = self->constructors[offset]->output_layer;
      self->play_frame->Y[offset] = UNIT_TO_BYTE(yuv[0]);
      self->play_frame->Cb[offset] = UNIT_TO_BYTE(yuv[1]);
      self->play_frame->Cr[offset] = UNIT_TO_BYTE(yuv[2]);
    }
  }
  fill_from_planar_u8(frame, self->play_frame->Y, RNNCA_WIDTH, RNNCA_HEIGHT);
}


static GstFlowReturn
gst_rnnca_transform_frame_ip(GstVideoFilter *filter,
    GstVideoFrame *frame)
{
  GstRnnca *self = GST_RNNCA(filter);
  GstFlowReturn ret = GST_FLOW_OK;
  if (self->training){
    GST_DEBUG("training");
    remember_frame(self, frame);
    maybe_learn(self);
  }
  if (self->playing){
    GST_DEBUG("playing");
    fill_frame(self, frame);
  }
  GST_LOG("rnnca_transform returning %d", ret);
  return ret;
}



static gboolean
plugin_init (GstPlugin * plugin)
{
  gboolean rnnca = gst_element_register(plugin, "rnnca", GST_RANK_NONE,\
      GST_TYPE_RNNCA);
  return rnnca;
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    rnnca,
    "Rnn cellular automata",
    plugin_init, VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
