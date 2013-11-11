#define PERIODIC_PGM_DUMP 1
#include "recur-nn.h"
#include "pgm_dump.h"
#include <math.h>
#include "path.h"
#include "badmaths.h"
#include <stdio.h>
#include <fenv.h>
#include <ctype.h>

#define NET_LOG_FILE "bptt.log"
#define BPTT_DEPTH 11
#define CONFAB_SIZE 70
#define LEARN_RATE 0.003
#define BIAS 1
#define HIDDEN_SIZE 19
#define INPUT_SIZE 2
#define BPTT_BATCH_SIZE 1

#define MOMENTUM 0.5
#define MOMENTUM_WEIGHT 1.0
#define FB1 7
#define FB2 3
#define K_STOP 200

static const char CONFAB_LUT[] = " |S$";
//static const char CONFAB_LUT[] = " /\\X";

static inline void
load_char_input(RecurNN *net, int c){
  if (net->bias)
    net->input_layer[0] = 1.0f;
  net->real_inputs[0] = (c & 1);
  net->real_inputs[1] = (c & 2) >> 1;
}

static inline float
net_error_bptt(RecurNN *net, float *error, int c, int next){
  load_char_input(net, c);
  float *answer = rnn_opinion(net, NULL);
  error[0] = (next & 1) - (answer[0] > 0);
  error[1] = (!!(next & 2)) - (answer[1] > 0);
  return (fabsf(error[0]) + fabsf(error[1])) * 0.5;
}

static float
sgd_one(RecurNN *net, const int current, const int next){
  RecurNNBPTT *bptt = net->bptt;
  bptt_advance(net);
  float sum = net_error_bptt(net, bptt->o_error, current, next);
  bptt_calculate(net);
  return sum;
}

static inline int
char_opinion(RecurNN *net, int c){
  load_char_input(net, c);
  float * answer = rnn_opinion(net, NULL);
  int a = ((answer[1] > 0) << 1) | (answer[0] > 0);
  return a;
}

static int
confabulate(RecurNN *net, char *text, int len, uint c){
  int i;
  c = MAX(c, 3);
  for (i = 0; i < len; i++){
    c = char_opinion(net, c);
    text[i] = CONFAB_LUT[c];
  }
  return text[i];
}

#define FIZZBUZZ(x, a, b) ((((x) % (a) == 0) << 1) + ((x) % (b) == 0));

void
epoch(RecurNN *net, const int len){
  int i;
  char confab[CONFAB_SIZE + 1];
  confab[CONFAB_SIZE] = 0;
  int current;
  int next = FIZZBUZZ(0, FB1, FB2);
  for(i = 1; i < len; i++){
    current = next;
    next = FIZZBUZZ(i, FB1, FB2);
    float error = sgd_one(net, current, next);
    net->bptt->learn_rate *= 0.999999;
    if ((i & 1023) == 0){
      int k = i >> 10;
      confabulate(net, confab, CONFAB_SIZE, next);
      DEBUG("%3dk %.2f %.4f confab: '%s'", k, error, net->bptt->learn_rate, confab);
      if (PERIODIC_PGM_DUMP){
        RecurNNBPTT *bptt = net->bptt;
        dump_colour_weights_autoname(net->ih_weights, net->h_size, net->i_size,
            "ih-k", net->generation);
        dump_colour_weights_autoname(net->ih_weights, net->h_size, net->hidden_size,
            "hh-k", net->generation);
        dump_colour_weights_autoname(net->ho_weights, net->o_size, net->h_size,
            "ho-k", net->generation);
        dump_colour_weights_autoname(bptt->ih_momentum, net->h_size, net->i_size,
            "ih-momentum-k", net->generation);
        dump_colour_weights_autoname(bptt->ih_delta, net->h_size, net->i_size,
            "ih-tmp-k", net->generation);
        dump_colour_weights_autoname(bptt->ho_momentum, net->o_size, net->h_size,
            "ho-momentum-k", net->generation);
      }
      if (K_STOP && k > K_STOP)
        exit(0);
    }
  }
}

void dump_parameters(void){
  DEBUG(
      "HIDDEN_SIZE %d\n"
      "LEARN_RATE %f\n"
      "BIAS %d\n"
      "MOMENTUM %f\n"
      ,
      HIDDEN_SIZE,
      LEARN_RATE,
      BIAS,
      MOMENTUM);
}

int
main(void){
  dump_parameters();
  feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
  RecurNN *net = rnn_new(INPUT_SIZE, HIDDEN_SIZE,
      INPUT_SIZE, BIAS ? RNN_NET_FLAG_STANDARD : RNN_NET_FLAG_NO_BIAS,
      1, NET_LOG_FILE, BPTT_DEPTH, LEARN_RATE, MOMENTUM, MOMENTUM_WEIGHT,
      BPTT_BATCH_SIZE, 1);
  START_TIMER(epoch);
  epoch(net, 5000000);
  DEBUG_TIMER(epoch);
  rnn_delete_net(net);
}
