#define DISABLE_PGM_DUMP 0
#define EXCESSIVE_PGM_DUMP 0
#define PERIODIC_PGM_DUMP 0
#define TEMPORAL_PGM_DUMP 0
#define CONFAB_HIDDEN_IMG 0
#define PERIODIC_SAVE_NET 1
#define TRY_RELOAD 0

#define NET_LOG_FILE "bptt.log"
#include "test-common.h"
#include "badmaths.h"
#include <stdio.h>
#include <fenv.h>
#include <ctype.h>

#define BPTT_DEPTH 20
#define CONFAB_SIZE 80
#define LEARN_RATE 0.001
#define LEARN_RATE_DECAY 0.96

#define MOMENTUM 0.95
#define MOMENTUM_WEIGHT 0.5
#define BIAS 1
#define K_STOP 0
#define BPTT_BATCH_SIZE 1

#define SAFE_CONFAB 1

u8 CHAR_TO_NET[257];
const u8 NET_TO_CHAR[] = "abcdefghijklmnopqrstuvwxyz,'- .\"#";

#define HIDDEN_SIZE 200
#define INPUT_SIZE (sizeof(NET_TO_CHAR) - 1)

#define NET_FILENAME ("test_backprop-" QUOTE(HIDDEN_SIZE) "-" QUOTE(BIAS) ".net")

static int
create_char_lut(u8 *ctn, const u8 *ntc){
  int i;
  int len = strlen((char *)ntc);
  int hash = strchr((char *)ntc, '#') - (char *)ntc;
  int space = strchr((char *)ntc, ' ') - (char *)ntc;
  memset(ctn, space, 257);
  for (i = 33; i < 127; i++){
    ctn[i] = hash;
  }
  for (i = 0; i < len; i++){
    u8 c = ntc[i];
    ctn[c] = i;
    if (islower(c))
      ctn[c - 32] = i;
  }
  return len;
}

static inline u8*
alloc_and_collapse_text(char *filename, long *len){
  int i, j;
  FILE *f = fopen(filename, "r");
  int err = fseek(f, 0, SEEK_END);
  *len = ftell(f);
  err |= fseek(f, 0, SEEK_SET);
  u8 *text = malloc(*len + 1);
  u8 prev = 0;
  u8 c;
  int chr = 0;
  int space = CHAR_TO_NET[' '];
  j = 0;
  for(i = 0; i < *len && chr != EOF; i++){
    chr = getc(f);
    c = CHAR_TO_NET[chr];
    if (c != space || prev != space){
      prev = c;
      text[j] = c;
      j++;
    }
  }
  text[j] = 0;
  *len = j;
  DEBUG("original text was %d chars, collapsed is %d", i, j);
  err |= fclose(f);
  if (err){
    DEBUG("something went wrong with the file %p (%s). error %d",
    f, filename, err);
  }
  return text;
}

static void UNUSED
dump_collapsed_text(u8 *text, int len, char *name)
{
  int i;
  FILE *f = fopen(name, "w");
  for (i = 0; i < len; i++){
    fputc(NET_TO_CHAR[text[i]], f);
  }
  fclose(f);
}


static inline float*
one_hot_opinion(RecurNN *net, const int hot){
  memset(net->real_inputs, 0, INPUT_SIZE * sizeof(float));
  net->real_inputs[hot] = 1.0f;
  float *answer = rnn_opinion(net, NULL);
  //net->real_inputs[hot] = 0.0;
  return answer;
}

static inline float
net_error_bptt(RecurNN *net, float *restrict error, int c, int next, int *correct){
  ASSUME_ALIGNED(error);
  float *answer = one_hot_opinion(net, c);
  int winner;
  float err = softmax_best_guess(error, answer, net->output_size, next, &winner);
  *correct = winner == next;
  return err;
}

static void
sgd_one(RecurNN *net, const int current, const int next, float *error, int *correct){
  RecurNNBPTT *bptt = net->bptt;
  float sum;
  bptt_advance(net);
  sum = net_error_bptt(net, bptt->o_error, current, next, correct);

  bptt_calculate(net);

  if (EXCESSIVE_PGM_DUMP && 0){
    dump_colour_weights_autoname(net->ho_weights, net->o_size, net->h_size,
        "ho", net->generation);
  }
  *error = sum;
}


static inline int
opinion_deterministic(RecurNN *net, int hot){
  float *answer = one_hot_opinion(net, hot);
  return search_for_max(answer, net->output_size);
}

static inline int
opinion_probabilistic(RecurNN *net, int hot){
  int i;
  float r;
  float *answer = one_hot_opinion(net, hot);
  float error[net->output_size];
  softmax(error, answer, net->output_size);

  /*outer loop in case error doesn't quite add to 1 */
  for(;;){
    r = rand_double(&net->rng);
    float accum = 0.0;
    for (i = 0; i < net->output_size; i++){
      accum += error[i];
      if (r < accum)
        return i;
    }
  }
}

static int
confabulate(RecurNN *net, char *text, int len, int c,
    int hidden_ppm, int deterministic){
  int i;
#if SAFE_CONFAB
  float tmp_hiddens[net->h_size];
  float tmp_inputs[net->i_size];
  memcpy(tmp_hiddens, net->hidden_layer, sizeof(tmp_hiddens));
  memcpy(tmp_inputs, net->input_layer, sizeof(tmp_inputs));
#endif
  float *im;
  if (hidden_ppm){
    im = malloc_aligned_or_die(net->h_size * len * sizeof(float));
  }
  if (c < 0 || c > 255)
    c = ' ';
  int n = CHAR_TO_NET[c];
  for (i = 0; i < len; i++){
    if (deterministic)
      n = opinion_deterministic(net, n);
    else
      n = opinion_probabilistic(net, n);
    c = NET_TO_CHAR[n];
    text[i] = c;
    if (hidden_ppm)
      memcpy(&im[net->h_size * i], net->hidden_layer, net->h_size * sizeof(float));
  }
  if (hidden_ppm){
    dump_colour_weights_autoname(im, net->h_size, len,
      "confab-hiddens", net->generation);
    free(im);
  }
#if SAFE_CONFAB
  memcpy(net->hidden_layer, tmp_hiddens, sizeof(tmp_hiddens));
  memcpy(net->input_layer, tmp_inputs, sizeof(tmp_inputs));
#endif
  return c;
}

static inline int
long_confab(RecurNN *net, int len, int rows, int c, int hidden_ppm){
  int i, j;
  char confab[len * rows + 1];
  confab[len * rows] = 0;
  c = confabulate(net, confab, len * rows, c, hidden_ppm, 0);
  for (i = 1; i < rows; i++){
    int target = i * len;
    int linebreak = target;
    int best = 100;
    for (j = MAX(target - 12, 1);
         j < MIN(target + 12, len * rows - 1); j++){
      if (confab[j] == ' '){
        int d = abs(j - target);
        if ( d < best){
          best = d;
          linebreak = j;
        }
      }
    }
    confab[linebreak] = '\n';
    if (best == 100){
      confab[linebreak - 1] = '}';
      confab[linebreak + 1] = '{';
    }
  }
  DEBUG("%s", confab);
  return c;
}

static TemporalPPM *input_ppm;

void
epoch(RecurNN *net, const u8 *text, const int len){
  int i;
  char confab[CONFAB_SIZE + 1];
  confab[CONFAB_SIZE] = 0;
  float error = 0.0f;
  float entropy = 0.0f;
  int correct = 0;
  for(i = 0; i < len - 1; i++){
    float e;
    int c;
    sgd_one(net, text[i], text[i + 1], &e, &c);
    correct += c;

    error += e;
    if (e < 1.0f)
      entropy += log2f(1.0f - e);
    else
      entropy -= 50;

    if (TEMPORAL_PGM_DUMP){
      temporal_ppm_add_row(input_ppm, net->input_layer);
    }

    if ((net->generation & 1023) == 0){
      int k = net->generation >> 10;
      entropy /= -1024.0f;
      confabulate(net, confab, CONFAB_SIZE, text[i], CONFAB_HIDDEN_IMG, 1);
      DEBUG("%4dk .%02d %.2f .%02d |%s|", k, (int)(error / 10.24f + 0.5), entropy,
          (int)(correct / 10.24f + 0.5), confab);
      bptt_log_float(net, "error", error / 1024.0f);
      bptt_log_float(net, "entropy", entropy);
      bptt_log_float(net, "accuracy", correct / 1024.0f);
      correct = 0;
      error = 0.0f;
      entropy = 0.0f;
      if (PERIODIC_SAVE_NET){
        rnn_save_net(net, NET_FILENAME);
      }
      if (PERIODIC_PGM_DUMP){
        rnn_multi_pgm_dump(net, "ihw ihm how hom");
      }
      if (K_STOP && k > K_STOP)
        exit(0);
    }
  }
  long_confab(net, CONFAB_SIZE, 6, ' ', CONFAB_HIDDEN_IMG);
}

void dump_parameters(void){
#define DEBUG_CONSTANT(x, f) DEBUG("%-25s: " #f, #x, x)
  DEBUG_CONSTANT(MAX_ERROR_GAIN , %g);
  DEBUG_CONSTANT(MIN_ERROR_FACTOR , %g);
  DEBUG_CONSTANT(HIDDEN_SIZE , %d);
  DEBUG_CONSTANT(LEARN_RATE , %g);
  DEBUG_CONSTANT(BIAS , %d);
  DEBUG_CONSTANT(MOMENTUM , %g);
  DEBUG_CONSTANT(MOMENTUM_WEIGHT , %g);
  DEBUG_CONSTANT(BPTT_DEPTH , %d);
  DEBUG_CONSTANT(BPTT_BATCH_SIZE, %d);
#undef DEBUG_CONSTANT
}

int
main(void){
  dump_parameters();
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  //feenableexcept(FE_ALL_EXCEPT & ~ FE_INEXACT);
  RecurNN *net = NULL;
#if TRY_RELOAD
  net = rnn_load_net(NET_FILENAME);
  if (net)
    rnn_set_log_file(net, NET_LOG_FILE);
  DEBUG("net is %p", net);
#endif
  if (net == NULL)
    net = rnn_new(INPUT_SIZE, HIDDEN_SIZE,
        INPUT_SIZE, BIAS ? RNN_NET_FLAG_STANDARD : RNN_NET_FLAG_NO_BIAS, 1,
        NET_LOG_FILE, BPTT_DEPTH, LEARN_RATE, MOMENTUM, MOMENTUM_WEIGHT,
        BPTT_BATCH_SIZE);

  create_char_lut(CHAR_TO_NET, NET_TO_CHAR);
  long len;
  u8* text = alloc_and_collapse_text(SRC_TEXT, &len);
  if (TEMPORAL_PGM_DUMP){
    input_ppm = temporal_ppm_alloc(net->i_size, 500, "input_layer", 0);
  }
  START_TIMER(run);
  for (int i = 0; i < 200; i++){
    DEBUG("Starting epoch %d. learn rate %f.", i, net->bptt->learn_rate);
    START_TIMER(epoch);
    epoch(net, text, len);
    DEBUG_TIMER(epoch);
    net->bptt->learn_rate *= LEARN_RATE_DECAY;
  }
  free(text);
  rnn_delete_net(net);
  free_temporal_ppm(input_ppm);
}
