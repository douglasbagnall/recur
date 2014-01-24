#include "recur-context.h"
#include "context-helpers.h"
#include "recur-common.h"
#include "recur-nn.h"
#include "rescale.h"
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

GST_DEBUG_CATEGORY_EXTERN(recur_context_debug);
#define GST_CAT_DEFAULT recur_context_debug

static inline float*
copy_audio_inputs(RecurNN *net, RecurContext *context)
{
  memcpy(net->real_inputs, context->current_audio, RECUR_N_MFCCS * sizeof(float));
  return net->real_inputs + RECUR_N_MFCCS;
}

static inline bool
scan_mask(u8* mask, int stride, int xpos, int ypos, int w, int h){
  for (int y = ypos; y < ypos + h; y++){
    for (int x = xpos; x < xpos + w; x++){
      if (mask[y * stride + x])
        return false;
    }
  }
  return true;
}

static inline void
fill_mask(u8* mask, int stride, int xpos, int ypos, int w, int h, uint colour){
  colour = MIN(colour, 255);
  for (int y = ypos; y < ypos + h; y++){
    memset(mask + stride * y + xpos, colour, w);
  }
}

static void
setup_trainers(RecurContext *context){
  /* just plonk trainers around randomly for now, using a mask to prevent
     overlaps */
  int m_width = RECUR_WORKING_WIDTH;
  int m_height = RECUR_WORKING_HEIGHT;
  u8* mask = calloc(m_width * m_height, 1);
  int scale_max;
  rand_ctx *rng = &context->net->rng;
  context->training_nets = rnn_new_training_set(context->net, RECUR_N_TRAINERS);
  for (scale_max = 5; scale_max; scale_max--){
    for (int j = 0, i = 0; i < RECUR_N_TRAINERS * 10; i++) {
      int scale = rand_small_int(rng, scale_max) + 1;
      int h = scale * RECUR_OUTPUT_HEIGHT;
      int w = scale * RECUR_OUTPUT_WIDTH;
      int margin = 2 * scale;
      int x = margin + rand_small_int(rng, m_width - w - 2 * margin);
      int y = margin + rand_small_int(rng, m_height - h - 2 * margin);
      if (scan_mask(mask, m_width, x, y, w, h)){
        GST_LOG("x %d, y %d, w %d, h %d, scale %d, i %d, j %d",
            x, y, w, h, scale, i, j);
        fill_mask(mask, m_width, x, y, w, h, 100 + 16 * j);
        context->trainers[j].x = x;
        context->trainers[j].y = y;
        context->trainers[j].scale = scale;
        context->trainers[j].net = context->training_nets[j];
        j++;
        if (j == RECUR_N_TRAINERS){
          goto done;
        }
      }
    }
    memset(mask, 0, m_width * m_height);
    GST_INFO("Couldn't fit in training nets with scale_max %d", scale_max);
  }

  GST_ERROR("Couldn't fit in training nets AT ALL!");/*XXX error handling*/
 done:
  pgm_dump(mask, m_width, m_height, IMAGE_DIR "mask.pgm");
  free(mask);
}


void
recur_setup_nets(RecurContext *context, const char *log_file)
{
  RecurNN *net = NULL;
  u32 flags = RNN_NET_FLAG_STANDARD | RNN_NET_FLAG_OWN_ACCUMULATORS;
#if TRY_RELOAD
  net = rnn_load_net(NET_FILENAME);
  DEBUG("net is %p", net);
#endif

  if (net == NULL){
    net = rnn_new(RECUR_N_MFCCS + RECUR_N_VIDEO_FEATURES,
        RECUR_N_HIDDEN, RECUR_OUTPUT_SIZE, flags, RECUR_RNG_SEED,
        log_file, RECUR_BPTT_DEPTH, LEARN_RATE, MOMENTUM);
    rnn_randomise_weights_auto(net);
  }
  context->net = net;
  setup_trainers(context);

  flags &= ~(RNN_NET_FLAG_OWN_WEIGHTS | RNN_NET_FLAG_OWN_BPTT);
  for (int i = 0; i < RECUR_N_CONSTRUCTORS; i++){
    context->constructors[i] = rnn_clone(net, flags, RECUR_RNG_SUBSEED, NULL);
  }
}

/*fill_video_nodes scales the u8 YCbCr planes of a RecurFrame down to
  w, h size planes of [0-1) float values */
static inline void
fill_video_nodes(float *dest, RecurFrame *frame, int w, int h,
    int xpos, int ypos, int scale){
  recur_integer_downscale_to_float(frame->Y, dest, RECUR_WORKING_WIDTH,
      xpos, ypos, w, h, scale);
  dest += w * h;
  recur_integer_downscale_to_float(frame->Cb, dest, RECUR_WORKING_WIDTH,
      xpos, ypos, w, h, scale);
  dest += w * h;
  recur_integer_downscale_to_float(frame->Cr, dest, RECUR_WORKING_WIDTH,
      xpos, ypos, w, h, scale);
}


static void
consolidate_and_apply_learning(RecurContext *context){
  /*XXX nets doesn't change, should be set at start up */
  rnn_apply_learning(context->net, RNN_MOMENTUM_WEIGHTED, 0,
      context->net->bptt->ih_accumulator, context->net->bptt->ho_accumulator);
}

void
recur_train_nets(RecurContext *context, RecurFrame *src_frame,
    RecurFrame *target_frame){

  for (int j = 0; j < RECUR_N_TRAINERS; j++){
    RecurTrainer *t = &context->trainers[j];
    RecurNN *net = t->net;
    rnn_bptt_advance(net);

    float *video_in = copy_audio_inputs(net, context);

    fill_video_nodes(video_in, src_frame,
        RECUR_INPUT_WIDTH + 2, RECUR_INPUT_HEIGHT + 2, t->x - t->scale, t->y - t->scale,
        t->scale * RECUR_RESOLUTION_GAIN);

    float *answer = rnn_opinion(net, NULL);
    ASSUME_ALIGNED(answer);
    fast_sigmoid_array(answer, answer, net->o_size);

    fill_video_nodes(net->bptt->o_error, target_frame,
        RECUR_OUTPUT_WIDTH, RECUR_OUTPUT_HEIGHT, t->x, t->y,
        t->scale);

    for (int i = 0; i < net->o_size; i++){
      float target = net->bptt->o_error[i];
      float a = answer[i];
      float slope = a * (1.0f - a);
      net->bptt->o_error[i] = slope * (target - a);
    }
    rnn_bptt_calc_deltas(net, net->bptt->ih_delta, net->bptt->ho_delta, NULL, NULL);
  }
  consolidate_and_apply_learning(context);

  rnn_condition_net(context->net);
  rnn_log_net(context->net);
}

void
possibly_save_state(RecurContext *context)
{
  RecurNN *net = context->net;
  if (PERIODIC_SAVE_NET && (net->generation & 1023) == 0){
    rnn_save_net(net, NET_FILENAME, 1);
    DEBUG("in possibly_save_state with generation %d", context->net->generation);
  }
  if (PERIODIC_PGM_DUMP && net->generation % PERIODIC_PGM_DUMP == 0){
    rnn_multi_pgm_dump(net, "hhw ihw");
  }
}

/*XXX unswizzle is for gain 2 only */
static inline void
unswizzle(int i, int *x, int *y)
{
  /* x is even bits, y is odd bits.
     unswizzling shuffle appropriate up to quite a big number */
  *x = i & 0x111;
  *x |= (i & 0x444) >> 1;
  *x = (*x & 3)    | ((*x & 0xffc) >> 2);
  *x = (*x & 15)   | ((*x & 0xff0) >> 2);
  *x = (*x & 0x3f) | ((*x & 0xfc0) >> 2);

  *y = (i & 0x222) >> 1;
  *y |= (i & 0x888) >> 2;
  *y = (*y & 3)    | ((*y & 0xffc) >> 2);
  *y = (*y & 0xf)  | ((*y & 0xff0) >> 2);
  *y = (*y & 0x3f) | ((*y & 0xfc0) >> 2);
}

static inline void
fill_sub_net_inputs(RecurContext *context, RecurNN *net, float *image, int left, int top){
  float *dest = copy_audio_inputs(net, context);
  int x_offset = RECUR_INPUT_WIDTH * left;
  int y_offset = RECUR_INPUT_HEIGHT * top;
  float *src = image;
  GST_LOG("left %d top %d x_offset is %d y_offset is %d"
      " sub image[0] is %f", left, top, x_offset, y_offset, *src);
  for (int i = 0; i < 3; i++){
    for (int y = y_offset - 1; y <= y_offset + RECUR_INPUT_HEIGHT; y++){
      int yy;
      if (y < 0){
        yy = RECUR_OUTPUT_HEIGHT - 1;
      }
      else if ( y >= RECUR_OUTPUT_HEIGHT){
        yy = 0;
      }
      else {
        yy = y;
      }
      for (int x = x_offset - 1; x <= x_offset + RECUR_INPUT_WIDTH; x++){
        int xx;
        if (x < 0)
          xx = RECUR_OUTPUT_WIDTH - 1;
        else if (x >= RECUR_OUTPUT_WIDTH)
          xx = 0;
        else
          xx = x;
        *dest = fast_sigmoid(src[yy * RECUR_OUTPUT_WIDTH + xx]);
        dest++;
      }
    }
    src += RECUR_OUTPUT_WIDTH * RECUR_OUTPUT_HEIGHT;
  }
}

static void
rnn_recursive_opinion(RecurContext *context, int index)
{
  int i;
  RecurNN **constructors = context->constructors;
  RecurNN *net = constructors[index];
  float *image = rnn_opinion(net, NULL);
  const int mul = RECUR_RESOLUTION_GAIN * RECUR_RESOLUTION_GAIN;
  int first_child = index * mul + 1;
  if (first_child < RECUR_N_CONSTRUCTORS){
    for (i = 0; i < mul; i++){
      int offset = first_child + i;
      net = constructors[offset];
      GST_LOG("net %d + %d, hiddens is %p, inputs %p",
          first_child, i, net->hidden_layer, net->input_layer);
      fill_sub_net_inputs(context, net, image,
          i % RECUR_RESOLUTION_GAIN,
          (i / RECUR_RESOLUTION_GAIN) % RECUR_RESOLUTION_GAIN);
      rnn_recursive_opinion(context, offset);
    }
  }
}

void
rnn_recursive_construct(RecurContext *context, u8 *Y, u8 *Cb, u8 *Cr,
    float *seed)
{
  int i;
  RecurNN *net = context->constructors[0];
  float * video_in = copy_audio_inputs(net, context);
  fast_sigmoid_array(video_in,
      seed, RECUR_N_VIDEO_FEATURES);

  GST_LOG("recursive construction starts");
  rnn_recursive_opinion(context, 0);
  /*
        0  1  4  5 16 17 20 21  64..
        2  3  6  7 18 19 22 23
        8  9 12 13 24 25 28 29
       10 11 14 15 26 27 30 31
       32 33 36 37
       34 35
  */
  int ow = RECUR_OUTPUT_WIDTH;
  int oh = RECUR_OUTPUT_HEIGHT;

  RecurNN **leaf_nets = context->constructors +\
    RECUR_N_CONSTRUCTORS - RECUR_CONSTRUCTOR_N_LEAVES;
  int last_gen_n = RECUR_CONSTRUCTOR_N_LEAVES;
  /*sqrt of trunk_net_n */
  int stride = RECUR_CONSTRUCTOR_WIDTH;

  for (i = 0; i < last_gen_n; i++){
    int x_pos, y_pos;
    unswizzle(i, &x_pos, &y_pos);
    net = leaf_nets[i];
    float *o = net->output_layer;
    /**XXX prefetching **/

    int offset = y_pos * stride * oh + x_pos * ow;
    for (int y = 0; y < oh; y++){
      fast_sigmoid_byte_array(Y + offset + stride * y, o + y * ow, ow);
    }
    o += oh * ow;
    for (int y = 0; y < oh; y++){
      fast_sigmoid_byte_array(Cb + offset + stride * y, o + y * ow, ow);
    }
    o += oh * ow;
    for (int y = 0; y < oh; y++){
      fast_sigmoid_byte_array(Cr + offset + stride * y, o + y * ow, ow);
    }
  }
}

void
recur_confabulate(RecurContext *context, u8 *Y, u8 *Cb, u8 *Cr){
  RecurNN *net = context->constructors[0];
  /*convert previous output into input (image scaling, audio?)
   *run the confab net */
  int i;
  float *dest = context->seed;
  float *src = net->output_layer;

  for (i = 0; i < 3; i++){
    /*convert first to working size */
    GST_LOG("dest %p, src %p, iw %d, ih %d, ow %d, oh %d",
        dest, src, RECUR_INPUT_WIDTH, RECUR_INPUT_HEIGHT,
        RECUR_OUTPUT_WIDTH, RECUR_OUTPUT_HEIGHT);

    recur_float_downscale(src, RECUR_OUTPUT_WIDTH, RECUR_OUTPUT_HEIGHT,
        RECUR_OUTPUT_WIDTH,
        dest, RECUR_INPUT_WIDTH, RECUR_INPUT_HEIGHT,
        RECUR_INPUT_WIDTH);
    dest += RECUR_INPUT_WIDTH * RECUR_INPUT_HEIGHT;
    src += RECUR_OUTPUT_WIDTH * RECUR_OUTPUT_HEIGHT;
  }
  rnn_recursive_construct(context, Y, Cb, Cr, context->seed);
}
