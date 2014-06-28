/* Copyright (C) 2014 Douglas Bagnall <douglas@halo.gen.nz> LGPL

This uses the RNN to predict the next character in a text sequence.
*/

#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include <math.h>
#include "path.h"
#include "badmaths.h"
#include <stdio.h>

#include "charmodel.h"
#include "utf8.h"

static inline float
capped_log2f(float x){
  return (x < 1e-30f) ? -100.0f : log2f(x);
}

/* rnn_char_find_alphabet returns the length of the found alphabet (not
   the same as strlen if utf8 is parsed), or -1 on failure. */
int
rnn_char_find_alphabet(const char *filename, int *alphabet, int *a_len,
    int *collapse_chars, int *c_len, double threshold, int ignore_case,
    int collapse_space, int utf8){
  int n_chars = utf8 ? 0x200000 : 256;
  int *counts = calloc(n_chars + 1, sizeof(int));
  int c, prev = 0;
  int n = 0;
  FILE *f = fopen_or_abort(filename, "r");
  /*alloc enough for all characters to be 4 bytes long */
  for(;;){
    c = fread_utf8_char(f);
    if (c < 0){
      STDERR_DEBUG("Unicode Error!");
      break;
    }
    else if (c == 0){
      break;
    }
    if (c == 31){/* 31 is metadata separator XXX gah */
      c = 32;
    }
    if (collapse_space){
      if (isspace(c)){
        c = 32;
        if (c == prev){
          continue;
        }
      }
    }
    if (ignore_case && c < 0x80){
      /*FIXME ascii only */
      if(isupper(c)){
        c = tolower(c);
      }
    }
    n++;
    counts[c]++;
    prev = c;
  }
  if (n == 0){
    goto error;
  }
  int a_count = 0;
  int c_count = 0;

  /*find the representative for the collapsed_chars, if any, which is put at
    the beginning of the alphabet.*/
  int max_collapsed_count = 0;
  int max_collapsed_point = 0;
  int min_count = MAX(threshold * n + 0.5, 1);
  DEBUG("min count %i threshold %f n %d", min_count, threshold, n);
  for (int i = 0; i < n_chars; i++){
    int count = counts[i];
    if (count && count < min_count){
      max_collapsed_count = count;
      max_collapsed_point = i;
    }
  }
  if (max_collapsed_count){
    alphabet[0] = max_collapsed_point;
    counts[max_collapsed_point] = 0; /*so as to ignore it hereafter*/
    a_count = 1;
  }
  /*map the rest of the collapsed chars to alphabet[0]*/
  for (int i = 0; i < n_chars; i++){
    int count = counts[i];
    if (count >= min_count){
      if (a_count == 256){
        goto error;
      }
      alphabet[a_count] = i;
      a_count++;
    }
    else if (count){
      if (c_count == 256){
        goto error;
      }
      collapse_chars[c_count] = i;
      c_count++;
    }
  }
  if (a_count == 0){
    goto error;
  }

  free(counts);
  *a_len = a_count;
  *c_len = c_count;
  DEBUG("a_len %i c_len %i", a_count, c_count);
  return 0;
 error:
  STDERR_DEBUG("threshold of %f over %d chars led to %d in alphabet, "
      "%d collapsed characters",
      threshold, n, a_count, c_count);
  free(counts);
  *a_len = *c_len = 0;
  return -1;
}

static int*
new_char_lut(const int *alphabet, int a_len, const int *collapse, int c_len,
    int *_space, int case_insensitive, int utf8){
  int i;
  int collapse_target = 0;
  int space = -1;
  for (i = 0; i < a_len; i++){
    if (alphabet[i] == ' '){
      space = i;
      break;
    }
  }
  if (space == -1){
    space = collapse_target;
    DEBUG("space is not in alphabet; using collapse_target");
  }
  *_space = space;
  int len = utf8 ? 0x200001 : 257;
  int *ctn = malloc(len *sizeof(int));
  /*anything unspecified goes to space */
  for (i = 0; i < len; i++){
    ctn[i] = space;
  }
  /*collapse chars map to alphabet[0] */
  for (i = 0; i < c_len; i++){
    int c = collapse[i];
    ctn[c] = collapse_target;
  }

  for (i = 0; i < a_len; i++){
    int c = alphabet[i];
    ctn[c] = i;
    /*FIXME: ascii only */
    if (islower(c) && case_insensitive){
      ctn[toupper(c)] = i;
    }
  }
  return ctn;
}

u8*
rnn_char_alloc_collapsed_text(char *filename, int *alphabet, int a_len,
    int *collapse_chars, int c_len, long *text_len,
    int case_insensitive, int collapse_space, int utf8, int quietness){
  int i, j;
  int space;
  int *char_to_net = new_char_lut(alphabet, a_len,
      collapse_chars, c_len, &space,
      case_insensitive, utf8);
  FILE *f = fopen_or_abort(filename, "r");
  int err = fseek(f, 0, SEEK_END);
  long len = ftell(f);
  err |= fseek(f, 0, SEEK_SET);
  u8 *text = malloc(len + 1);
  u8 prev = 0;
  u8 c;
  int chr = 0;
  j = 0;
  for(i = 0; i < len; i++){
    if (utf8){
      chr = fread_utf8_char(f);
      if (chr < 0){
        break;
      }
    }
    else {
      chr = fgetc(f);
      if (chr == EOF)
        break;
    }

    c = char_to_net[chr];
    if (collapse_space && (c != space || prev != space)){
      prev = c;
      text[j] = c;
      j++;
    }
    else {
      text[j] = c;
      j = i;
    }
  }
  text[j] = 0;
  *text_len = j;
  free(char_to_net);
  if (quietness < 1){
    STDERR_DEBUG("original text was %d chars (%d bytes), collapsed is %d",
        i, (int)len, j);
  }
  err |= fclose(f);
  if (err && quietness < 2){
    STDERR_DEBUG("something went wrong with the file %p (%s). error %d",
        f, filename, err);
  }
  return text;
}

void
rnn_char_dump_collapsed_text(const u8 *text, int len, const char *name,
    const char *alphabet)
{
  int i;
  FILE *f = fopen_or_abort(name, "w");
  for (i = 0; i < len; i++){
    u8 c = text[i];
    fputc(alphabet[c], f);
  }
  fclose(f);
}

static inline int
search_for_max(float *answer, int len){
  ASSUME_ALIGNED(answer);
  int j;
  int best_offset = 0;
  float best_score = *answer;
  for (j = 1; j < len; j++){
    if (answer[j] >= best_score){
      best_score = answer[j];
      best_offset = j;
    }
  }
  return best_offset;
}

static inline float*
one_hot_opinion(RecurNN *net, int hot){
  float *inputs;
  int len;
  if (net->bottom_layer){
    inputs = net->bottom_layer->inputs;
    len = net->bottom_layer->input_size;
  }
  else{
    inputs = net->real_inputs;
    len = net->input_size;
  }

  //XXX could just set the previous one to zero (i.e. remember it)
  memset(inputs, 0, len * sizeof(float));
  inputs[hot] = 1.0f;
  return rnn_opinion(net, NULL);
}

static float
net_error_bptt(RecurNN *net, float *restrict error, int c, int next, int *correct){
  ASSUME_ALIGNED(error);
  float *answer = one_hot_opinion(net, c);
  int winner;
  winner = softmax_best_guess(error, answer, net->output_size);
  *correct = (winner == next);
  error[next] += 1.0f;
  return error[next];
}


int
opinion_deterministic(RecurNN *net, int hot){
  float *answer = one_hot_opinion(net, hot);
  return search_for_max(answer, net->output_size);
}

int
opinion_probabilistic(RecurNN *net, int hot, float bias){
  int i;
  float r;
  float *answer = one_hot_opinion(net, hot);
  int n_chars = net->output_size;
  float error[net->output_size];
  biased_softmax(error, answer, n_chars, bias);
  /*outer loop in case error doesn't quite add to 1 */
  for(;;){
    r = rand_double(&net->rng);
    float accum = 0.0;
    for (i = 0; i < n_chars; i++){
      accum += error[i];
      if (r < accum)
        return i;
    }
  }
}

float
validate(RecurNN *net, const u8 *text, int len){
  float error[net->output_size];
  float entropy = 0.0f;
  int i;
  int n_chars = net->output_size;
  /*skip the first few because state depends too much on previous experience */
  int skip = MIN(len / 10, 5);
  for (i = 0; i < skip; i++){
    one_hot_opinion(net, text[i]);
  }
  for (; i < len - 1; i++){
    float *answer = one_hot_opinion(net, text[i]);
    softmax(error, answer, n_chars);
    float e = error[text[i + 1]];
    entropy += capped_log2f(e);
  }
  entropy /= -(len - skip - 1);
  return entropy;
}

static void
eval_simple(RnnCharSchedule *s, RecurNN *net, float score, int verbose){
  int i, j;
  RecurNNBPTT *bptt = net->bptt;
  if (bptt->learn_rate <= s->learn_rate_min){
    return;
  }
  int sample_size = s->recent_len / 3;
  i = rand_small_int(&net->rng, s->recent_len);
  s->recent[i] = score;
  if (s->timeout){
    s->timeout--;
    return;
  }
  for (++i, j = 0; j < sample_size; j++, i++){
    if (i >= s->recent_len)
      i = 0;
    if (score < s->recent[i]){
      return;
    }
  }
  s->timeout = s->recent_len;
  bptt->learn_rate = MAX(s->learn_rate_min, bptt->learn_rate * s->learn_rate_mul);
  if (verbose){
    DEBUG("generation %7d: entropy %.4g exceeds %d recent samples."
        " setting learn_rate to %.3g. momentum %.3g",
        net->generation, score, sample_size,
        bptt->learn_rate, net->bptt->momentum);
  }
}

void
rnn_char_init_schedule(RnnCharSchedule *s, int recent_len,
    float learn_rate_min, float learn_rate_mul){
  s->recent = malloc_aligned_or_die(recent_len * sizeof(float));
  s->recent_len = recent_len;
  s->learn_rate_min = learn_rate_min;
  s->learn_rate_mul = learn_rate_mul;
  for (int i = 0; i < s->recent_len; i++){
    s->recent[i] = 1e10;
  }
  s->timeout = s->recent_len;
  s->eval = eval_simple;
}



int
rnn_char_confabulate(RecurNN *net, char *dest, int char_len,
    int byte_len, const int* alphabet, int utf8, float bias){
  int i, j;
  static int n = 0;
  int safe_end = byte_len - (utf8 ? 5 : 1);
  for (i = 0, j = 0; i < char_len && j < safe_end; i++){
    if (bias > 100)
      n = opinion_deterministic(net, n);
    else
      n = opinion_probabilistic(net, n, bias);
    if (utf8){
      j += write_utf8_char(alphabet[n], dest + j);
    }
    else {
      j = i;
      dest[j] = alphabet[n];
    }
  }
  dest[j] = 0;
  return j;
}


void
rnn_char_init_ventropy(RnnCharVentropy *v, RecurNN *net, const u8 *text, const int len,
    const int lap){
  v->net = net;
  v->text = text;
  v->len = len;
  v->lap = lap;
  v->lapsize = len / lap;
  v->history = calloc(lap, sizeof(float));
  v->entropy = 0;
  v->counter = 0;
}

float
rnn_char_calc_ventropy(RnnCharModel *model, RnnCharVentropy *v, int lap)
{
  if (v->len > 0){
    if (v->lap > 1 && lap){
      v->counter++;
      if (v->counter == v->lap){
        v->counter = 0;
      }
      v->history[v->counter] = validate(v->net, v->text + v->lapsize * v->counter,
          v->lapsize);
      float sum = 0.0f;
      float div = v->lap;
      for (int j = 0; j < v->lap; j++){
        div -= v->history[j] == 0;
        sum += v->history[j];
      }
      v->entropy = div ? sum / div : 0;
    }
    else {
      v->entropy = validate(v->net, v->text, v->len);
      v->history[0] = v->entropy;
    }
  }
  return v->entropy;
}


int
rnn_char_epoch(RnnCharModel *model, RecurNN *confab_net, RnnCharVentropy *v,
    const u8 *text, const int len,
    const int start, const int stop,
    float confab_bias, int confab_size,
    int quietness){
  int i, j;
  float error = 0.0f;
  float entropy = 0.0f;
  int correct = 0;
  float e;
  int c;
  int n_nets = model->n_training_nets;
  int spacing = (len - 1) / n_nets;
  RecurNN *net = model->net;
  RecurNN **nets = model->training_nets;
  uint report_counter = net->generation % model->report_interval;
  struct timespec timers[2];
  struct timespec *time_start = timers;
  struct timespec *time_end = timers + 1;
  clock_gettime(CLOCK_MONOTONIC, time_start);
  for(i = start; i < len - 1; i++){
    float momentum = rnn_calculate_momentum_soft_start(net->generation,
        model->momentum, model->momentum_soft_start);
    if (n_nets > 1 || model->momentum_style != RNN_MOMENTUM_WEIGHTED ||
        model->use_multi_tap_path){
      for (j = 0; j < n_nets; j++){
        RecurNN *n = nets[j];
        int offset = i + j * spacing;
        if (offset >= len - 1){
          offset -= len - 1;
        }
        rnn_bptt_advance(n);
        e = net_error_bptt(n, n->bptt->o_error,
            text[offset], text[offset + 1], &c);
        correct += c;
        error += e;
        entropy += capped_log2f(1.0f - e);
        /*Second argument to r_b_c_deltas toggles delta accumulation. Turning
          it off on the first run avoids expicit zeroing outside of the loop
          (via rnn_bptt_clear_deltas) and is thus slightly faster.
         */
        rnn_bptt_calc_deltas(n, j ? 1 : 0);
      }
      rnn_apply_learning(net, model->momentum_style, momentum);
    }
    else {
      RecurNNBPTT *bptt = net->bptt;
      bptt->momentum = momentum;
      rnn_bptt_advance(net);
      e = net_error_bptt(net, bptt->o_error, text[i], text[i + 1], &c);
      rnn_bptt_calculate(net, model->batch_size);
      correct += c;
      error += e;
      entropy += capped_log2f(1.0f - e);
    }

    if (model->input_ppm){
      temporal_ppm_add_row(model->input_ppm, net->input_layer);
    }
    if (model->error_ppm){
      temporal_ppm_add_row(model->error_ppm, net->bptt->o_error);
    }
    report_counter++;
    if (report_counter >= model->report_interval){
      report_counter = 0;
      clock_gettime(CLOCK_MONOTONIC, time_end);
      s64 secs = time_end->tv_sec - time_start->tv_sec;
      s64 nano = time_end->tv_nsec - time_start->tv_nsec;
      double elapsed = secs + 1e-9 * nano;
      struct timespec *tmp = time_end;
      time_end = time_start;
      time_start = tmp;
      float ventropy = rnn_char_calc_ventropy(model, v, 1);

      /*formerly in report_on_progress*/
      {
        float scale = 1.0f / (model->report_interval * n_nets);
        int k = net->generation >> 10;
        entropy *= -scale;
        error *= scale;
        float accuracy = correct * scale;
        double per_sec = 1.0 / scale / elapsed;
        if (confab_net && confab_size && quietness < 1){
          int alloc_size = confab_size * 4;
          char confab[alloc_size + 1];
          rnn_char_confabulate(confab_net, confab, confab_size, alloc_size,
              model->alphabet, model->utf8, confab_bias);
          STDERR_DEBUG("%5dk e.%02d t%.2f v%.2f a.%02d %.0f/s |%s|", k,
              (int)(error * 100 + 0.5),
              entropy, ventropy,
              (int)(accuracy * 100 + 0.5), per_sec + 0.5, confab);

        }
        rnn_log_float(net, "t_error", error);
        rnn_log_float(net, "t_entropy", entropy);
        rnn_log_float(net, "v_entropy", ventropy);
        rnn_log_float(net, "momentum", net->bptt->momentum);
        rnn_log_float(net, "accuracy", accuracy);
        rnn_log_float(net, "learn-rate", net->bptt->learn_rate);
        rnn_log_float(net, "per_second", per_sec);
        correct = 0;
        error = 0.0f;
        entropy = 0.0f;
      }

      if (model->save_net && model->filename){
        rnn_save_net(net, model->filename, 1);
      }
      if (model->periodic_pgm_dump_string){
        rnn_multi_pgm_dump(net, model->periodic_pgm_dump_string,
            model->pgm_name);
      }
      model->schedule.eval(&model->schedule, net, ventropy, model->quiet < 2);
      if (model->periodic_weight_noise){
        rnn_weight_noise(net, model->periodic_weight_noise);
      }
    }
    if (stop && (int)net->generation >= stop){
      return 1;
    }
  }
  return 0;
}

char *
rnn_char_construct_metadata(const struct RnnCharMetadata *m){
  char *metadata;
  int ret = asprintf(&metadata,
#define SEP "\x1F"
      "alphabet"       SEP "%s" SEP
      "collapse_chars" SEP "%s"
#undef SEP
      ,
      m->alphabet,
      m->collapse_chars
  );
  if (ret == -1){
    FATAL_ERROR("can't alloc memory for metadata. or something.");
  }
  return metadata;
}

int
rnn_char_load_metadata(const char *metadata, struct RnnCharMetadata *m){

  /*0x1f is the ascii field separator character.*/

#define CHECK_KEY(str, wanted) do {                                     \
    char * token = strtok(str, "\x1F");                                 \
    if (strcmp(token, wanted)){                                         \
      STDERR_DEBUG("looking for '%s', found '%s'", wanted, token);      \
      goto error;                                                       \
    }                                                                   \
  }while(0)                                                             \

  char *s = strdup(metadata);
  CHECK_KEY(s, "alphabet");
  m->alphabet = strdup(strtok(NULL, "\x1F"));
  CHECK_KEY(s, "collapse_chars");
  m->collapse_chars = strdup(strtok(NULL, "\x1F"));

#undef CHECK_KEY

  free(s);
  return 0;
 error:
  return 1;
}

void
rnn_char_free_metadata_items(struct RnnCharMetadata *m){
  free(m->alphabet);
  free(m->collapse_chars);
}

char*
rnn_char_construct_net_filename(struct RnnCharMetadata *m, const char *basename,
    int alpha_size, int bottom_size, int hidden_size){
  char s[260];
  char *metadata = rnn_char_construct_metadata(m);
  int input_size = alpha_size;
  int output_size = alpha_size;
  u32 sig = rnn_hash32(metadata);
  if (bottom_size){
    snprintf(s, sizeof(s), "%s-s%0" PRIx32 "-i%d-b%d-h%d-o%d.net", basename,
        sig, input_size, bottom_size, hidden_size, output_size);
  }
  else{
    snprintf(s, sizeof(s), "%s-s%0" PRIx32 "-i%d-h%d-o%d.net", basename,
        sig, input_size, hidden_size, output_size);
  }
  DEBUG("filename: %s", s);
  return strdup(s);
}
