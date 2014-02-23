#include "recur-nn.h"
#include <cdb.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

const char *FORMAT_VERSION = "save_format_version";

int
rnn_save_net(RecurNN *net, const char *filename, int backup){
  MAYBE_DEBUG("saving net at generation %d", net->generation);
  struct cdb_make cdbm;
  int fd = -1;
  char tmpfn[] = "tmp_net_XXXXXX";
  int ret = 0;
  if (net == NULL || filename == NULL)
    goto early_error;
  fd = mkostemp(tmpfn, O_RDWR | O_CREAT);
  if (fd == -1){
    DEBUG("cannot open temporary file for writing");
    perror();
    goto early_error;
  }
  ret = cdb_make_start(&cdbm, fd);
  if (ret)
    goto error;

  /* save a version number                */
  /* none (or 0,1): original version      */
  /* 2: saves ho_scale                    */
  /* 3: saves BPTT min_error_factor       */
  /* 4: saves bottom layer if applicable, using more qualified keys
     ("net->X", "bptt->X", "bottom_layer->X") */
  const int version = 4;
  cdb_make_add(&cdbm, FORMAT_VERSION, strlen(FORMAT_VERSION), &version, sizeof(version));

#define SAVE_SCALAR(obj, attr) do {                                     \
    char *key = (version >= 4) ? QUOTE(obj) "." QUOTE(attr) :          \
      QUOTE(attr);                                                      \
    ret = cdb_make_add(&cdbm, key, strlen(key),                         \
        &obj->attr, sizeof(obj->attr));                                 \
    if (ret){                                                           \
      DEBUG("error %d saving '%s'", ret, key);                          \
      goto error;                                                       \
    }} while (0)

#define SAVE_ARRAY(obj, attr, len) do {                                 \
    char *key = (version >= 4) ? QUOTE(obj) "." QUOTE(attr) :          \
      QUOTE(attr);                                                      \
    ret = cdb_make_add(&cdbm, key, strlen(key),                         \
        obj->attr, sizeof(*obj->attr) * len);                           \
    if (ret){                                                           \
      DEBUG("error %d saving '%s'", ret, key);                          \
      goto error;                                                       \
    }} while (0)
  /* not saved:
        bptt->ih_scale (temporary)
        net->real_inputs (pointer)
        net->bptt (pointer)
        net->log (file handle)
        bptt->mem (pointer for deallocation)
  */
  SAVE_SCALAR(net, i_size);
  SAVE_SCALAR(net, h_size);
  SAVE_SCALAR(net, o_size);
  SAVE_SCALAR(net, input_size);
  SAVE_SCALAR(net, hidden_size);
  SAVE_SCALAR(net, output_size);
  SAVE_SCALAR(net, ih_size);
  SAVE_SCALAR(net, ho_size);
  SAVE_SCALAR(net, bias);
  SAVE_SCALAR(net, generation);
  SAVE_SCALAR(net, flags);
  SAVE_SCALAR(net, rng); /* a struct, should work? */

  SAVE_ARRAY(net, input_layer, net->i_size);
  SAVE_ARRAY(net, hidden_layer, net->h_size);
  SAVE_ARRAY(net, output_layer, net->o_size);
  SAVE_ARRAY(net, ih_weights, net->ih_size);
  SAVE_ARRAY(net, ho_weights, net->ho_size);

  if ((net->flags & RNN_NET_FLAG_OWN_BPTT) && net->bptt){
    RecurNNBPTT *bptt = net->bptt;
    SAVE_SCALAR(bptt, depth);
    SAVE_SCALAR(bptt, index);
    SAVE_SCALAR(bptt, learn_rate);
    SAVE_SCALAR(bptt, ho_scale);   /*version 2 and above*/
    SAVE_SCALAR(bptt, momentum);
    SAVE_SCALAR(bptt, momentum_weight);
    SAVE_SCALAR(bptt, min_error_factor);   /*version 3 and above*/
    SAVE_ARRAY(bptt, i_error, net->i_size);
    SAVE_ARRAY(bptt, h_error, net->h_size);
    SAVE_ARRAY(bptt, o_error, net->o_size);
    SAVE_ARRAY(bptt, ih_momentum, net->ih_size);
    SAVE_ARRAY(bptt, ho_momentum, net->ho_size);
    SAVE_ARRAY(bptt, history, net->bptt->depth * net->i_size);
    SAVE_ARRAY(bptt, ih_delta, net->ih_size);
  }
  if (net->bottom_layer){
    /*version 4+ */
    RecurExtraLayer *bottom_layer = net->bottom_layer;
    int matrix_size = bottom_layer->i_size * bottom_layer->o_size;
    SAVE_SCALAR(bottom_layer, input_size);
    SAVE_SCALAR(bottom_layer, output_size);
    SAVE_SCALAR(bottom_layer, i_size);
    SAVE_SCALAR(bottom_layer, o_size);
    SAVE_SCALAR(bottom_layer, learn_rate_scale);
    SAVE_SCALAR(bottom_layer, overlap);
    SAVE_ARRAY(bottom_layer, weights, matrix_size);
  }
#undef SAVE_SCALAR
#undef SAVE_ARRAY

  cdb_make_finish(&cdbm);
  close(fd);
  if (backup){
    char *backup_filename;
    int size = asprintf(&backup_filename, "%s~", filename);
    if (size != -1){
      if (size == (int)strlen(filename + 2)){
        rename(filename, backup_filename);
      }
      free(backup_filename);
    }
  }
  /*XXX should check the rename errors */
  rename(tmpfn, filename);
  return 0;
 error:
  cdb_make_finish(&cdbm);
  close(fd);
 early_error:
  DEBUG("failed to save net %p with fd %d ret %d errno %d filename '%s'",
      net, fd, ret, errno, filename ? filename : "(nil, which is the problem)");
  return -1;
}

RecurNN*
rnn_load_net(const char *filename){
  int fd;
  uint vlen;
  int ret = 0;
  RecurNN tmpnet;
  RecurNNBPTT tmpbptt;
  RecurExtraLayer tmpbl;
  RecurNN *net = &tmpnet;
  RecurNNBPTT *bptt = &tmpbptt;
  RecurExtraLayer *bottom_layer = &tmpbl;

  fd = open(filename, O_RDONLY);
  if (fd == -1){
    DEBUG("can't open '%s' (%s)", filename, strerror(errno));
    goto open_error;
  }

  int version = 0;
  if (cdb_seek(fd, FORMAT_VERSION, strlen(FORMAT_VERSION), &vlen) >= 0){
    if (vlen == sizeof(version)){
      cdb_bread(fd, &version, vlen);
    }
  }
  /*gather up scalar values in tmpnet/tmpbptt*/
#define READ_SCALAR(obj, attr) do {\
    char *key = (version >= 4) ? QUOTE(obj) "." QUOTE(attr) : QUOTE(attr); \
    ret = cdb_seek(fd, key, strlen(key), &vlen);                        \
    if (ret < 1){ DEBUG("error %d loading '%s'", ret, key);             \
      goto pre_alloc_error;}                                            \
    if (vlen != sizeof(obj->attr)) {                                    \
      DEBUG("size mismatch on '%s->%s' want %u, found %lu",             \
          QUOTE(obj), QUOTE(attr), vlen, sizeof(obj->attr));            \
      goto pre_alloc_error;                                             \
    }                                                                   \
    cdb_bread(fd, &obj->attr, vlen);                                    \
  } while (0)

  READ_SCALAR(net, i_size);
  READ_SCALAR(net, h_size);
  READ_SCALAR(net, o_size);

  READ_SCALAR(net, input_size);
  READ_SCALAR(net, hidden_size);
  READ_SCALAR(net, output_size);

  READ_SCALAR(net, ih_size);
  READ_SCALAR(net, ho_size);
  READ_SCALAR(net, bias);
  READ_SCALAR(net, rng);
  READ_SCALAR(net, generation);
  READ_SCALAR(net, flags);
  if (tmpnet.flags & RNN_NET_FLAG_OWN_BPTT){
    READ_SCALAR(bptt, depth);
    READ_SCALAR(bptt, learn_rate);
    READ_SCALAR(bptt, index);
    READ_SCALAR(bptt, momentum);
    READ_SCALAR(bptt, momentum_weight);
    if (version >= 2){
      READ_SCALAR(bptt, ho_scale);
    }
    if (version >= 3){
      READ_SCALAR(bptt, min_error_factor);
    }
  }
  if ((tmpnet.flags & RNN_NET_FLAG_BOTTOM_LAYER) && version >= 4){
    READ_SCALAR(bottom_layer, learn_rate_scale);
    READ_SCALAR(bottom_layer, input_size);
    READ_SCALAR(bottom_layer, output_size);
    READ_SCALAR(bottom_layer, i_size);
    READ_SCALAR(bottom_layer, o_size);
    READ_SCALAR(bottom_layer, overlap);
  }
#undef READ_SCALAR

  if (tmpnet.flags & RNN_NET_FLAG_BOTTOM_LAYER){
    net = rnn_new_with_bottom_layer(tmpbl.input_size, tmpbl.output_size,
        tmpnet.hidden_size, tmpnet.output_size, tmpnet.flags, 0, NULL,
        tmpbptt.depth, tmpbptt.learn_rate, tmpbptt.momentum, tmpbl.overlap);
  }
  else {
    net = rnn_new(tmpnet.input_size, tmpnet.hidden_size,
      tmpnet.output_size, tmpnet.flags, 0, NULL,
      tmpbptt.depth, tmpbptt.learn_rate, tmpbptt.momentum);
  }
  bptt = net->bptt;
  bottom_layer = net->bottom_layer;
  net->rng = tmpnet.rng;
  net->generation = tmpnet.generation;

  if (bptt){
    bptt->index = tmpbptt.index;
    bptt->momentum_weight = tmpbptt.momentum_weight;
    if (version >= 2){
      bptt->ho_scale = tmpbptt.ho_scale;
    }
    if (version >= 3){
      bptt->min_error_factor = tmpbptt.min_error_factor;
    }
  }
#define CHECK_SCALAR(new, tmp, attr) do {                               \
    if (new->attr != tmp.attr){                                         \
      DEBUG("attribute '%s' differs %f vs %f", QUOTE(attr),             \
          (float)new->attr, (float)tmp.attr);                           \
      goto error;}                                                      \
  } while (0)

  CHECK_SCALAR(net, tmpnet, i_size);
  CHECK_SCALAR(net, tmpnet, h_size);
  CHECK_SCALAR(net, tmpnet, o_size);

  CHECK_SCALAR(net, tmpnet, input_size);
  CHECK_SCALAR(net, tmpnet, hidden_size);
  CHECK_SCALAR(net, tmpnet, output_size);

  CHECK_SCALAR(net, tmpnet, ih_size);
  CHECK_SCALAR(net, tmpnet, ho_size);
  CHECK_SCALAR(net, tmpnet, bias);

  CHECK_SCALAR(net, tmpnet, rng.a);
  CHECK_SCALAR(net, tmpnet, rng.b);
  CHECK_SCALAR(net, tmpnet, rng.c);
  CHECK_SCALAR(net, tmpnet, rng.d);

  CHECK_SCALAR(net, tmpnet, generation);
  CHECK_SCALAR(net, tmpnet, flags);
  if (bptt){
    CHECK_SCALAR(bptt, tmpbptt, depth);
    CHECK_SCALAR(bptt, tmpbptt, index);
    if (version >= 2){
      CHECK_SCALAR(bptt, tmpbptt, ho_scale);
    }
    else {
      /*ho_scale wasn't originally saved. But it was always set as follows. */
      bptt->ho_scale = ((float)tmpnet.output_size) / tmpnet.hidden_size;
    }
    if (version >= 3){
      CHECK_SCALAR(bptt, tmpbptt, min_error_factor);
    }
    else {
      bptt->min_error_factor = BASE_MIN_ERROR_FACTOR * net->h_size;
    }
  }

  if (net->bottom_layer && 0){
    CHECK_SCALAR(net->bottom_layer, tmpbl, input_size);
    CHECK_SCALAR(net->bottom_layer, tmpbl, output_size);
    CHECK_SCALAR(net->bottom_layer, tmpbl, i_size);
    CHECK_SCALAR(net->bottom_layer, tmpbl, o_size);
    CHECK_SCALAR(net->bottom_layer, tmpbl, learn_rate_scale);
    CHECK_SCALAR(net->bottom_layer, tmpbl, overlap);
  }
#undef CHECK_SCALAR

  /* so presumably all the arrays are all allocate and the right size. */
#define READ_ARRAY(obj, attr, size) do {                                \
    char *key = (version >= 4) ? QUOTE(obj) "." QUOTE(attr) : QUOTE(attr); \
    ret = cdb_seek(fd, key, strlen(key), &vlen);                        \
    if (ret < 1){ DEBUG("error %d loading '%s'", ret, key);             \
      goto error;}                                                      \
    if (vlen != size) {                                                 \
      DEBUG("array size mismatch on '%s->%s' saved %u, calculated %lu", \
          QUOTE(obj), QUOTE(attr), vlen, size);                         \
      goto error;                                                       \
    }                                                                   \
    cdb_bread(fd, obj->attr, size);                                     \
  } while (0)

  READ_ARRAY(net, input_layer, net->i_size * sizeof(float));
  READ_ARRAY(net, hidden_layer, net->h_size * sizeof(float));
  READ_ARRAY(net, output_layer, net->o_size * sizeof(float));
  READ_ARRAY(net, ih_weights, net->ih_size * sizeof(float));
  READ_ARRAY(net, ho_weights, net->ho_size * sizeof(float));

  if(bptt){
    READ_ARRAY(bptt, i_error, net->i_size * sizeof(float));
    READ_ARRAY(bptt, h_error, net->h_size * sizeof(float));
    READ_ARRAY(bptt, o_error, net->o_size * sizeof(float));
    READ_ARRAY(bptt, ih_momentum, net->ih_size * sizeof(float));
    READ_ARRAY(bptt, ho_momentum, net->ho_size * sizeof(float));
    READ_ARRAY(bptt, history, bptt->depth * net->i_size * sizeof(float));
    READ_ARRAY(bptt, ih_delta, net->ih_size * sizeof(float));
  }
  if (bottom_layer){
    READ_ARRAY(bottom_layer, weights, bottom_layer->i_size * bottom_layer->o_size * sizeof(float));
    /*not restoring full state (momentums etc) */
  }
#undef READ_ARRAY
  close(fd);
  DEBUG("successfully loaded net '%s'", filename);
  return net;
 error:
  rnn_delete_net(net);
 pre_alloc_error:
  close(fd);
 open_error:
  DEBUG("loading net failed!");
  return NULL;
}
