#include "recur-nn.h"
#include <cdb.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

int
rnn_save_net(RecurNN *net, const char *filename){
  MAYBE_DEBUG("saving net at generation %d", net->generation);
  struct cdb_make cdbm;
  int fd;
  char tmpfn[] = "tmp_net_XXXXXX";
  int ret = 0;

  fd = mkostemp(tmpfn, O_RDWR | O_CREAT);
  if (fd == -1)
    goto early_error;
  ret = cdb_make_start(&cdbm, fd);
  if (ret)
    goto error;

#define SAVE_SCALAR(obj, attr) do {                                     \
    ret = cdb_make_add(&cdbm, QUOTE(attr), strlen(QUOTE(attr)),         \
        &obj->attr, sizeof(obj->attr));                                 \
    if (ret){                                                           \
      DEBUG("error %d saving '%s'", ret, QUOTE(attr));                  \
      goto error;                                                       \
    }} while (0)

#define SAVE_ARRAY(obj, attr, len) do {                                 \
    ret = cdb_make_add(&cdbm, QUOTE(attr), strlen(QUOTE(attr)),         \
        obj->attr, sizeof(*obj->attr) * len);                           \
    if (ret){                                                           \
      DEBUG("error %d saving '%s'", ret, QUOTE(attr));                  \
      goto error;                                                       \
    }} while (0)
  //not saving bptt->scale
  //not saving pointers net->real_inputs or net->bptt
  // not saving net->log, bptt->mem
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

  SAVE_SCALAR(net->bptt, depth);
  SAVE_SCALAR(net->bptt, batch_size);
  SAVE_SCALAR(net->bptt, index);
  SAVE_SCALAR(net->bptt, learn_rate);
  SAVE_SCALAR(net->bptt, momentum);
  SAVE_SCALAR(net->bptt, momentum_weight);
  SAVE_ARRAY(net->bptt, i_error, net->i_size);
  SAVE_ARRAY(net->bptt, h_error, net->h_size);
  SAVE_ARRAY(net->bptt, o_error, net->o_size);
  SAVE_ARRAY(net->bptt, ih_momentum, net->ih_size);
  SAVE_ARRAY(net->bptt, ho_momentum, net->ho_size);
  SAVE_ARRAY(net->bptt, history, net->bptt->depth * net->i_size);
  SAVE_ARRAY(net->bptt, ih_delta, net->ih_size);

#undef SAVE_SCALAR
#undef SAVE_ARRAY

  cdb_make_finish(&cdbm);
  close(fd);
  rename(tmpfn, filename);
  return 0;
 error:
  cdb_make_finish(&cdbm);
  close(fd);
 early_error:
  DEBUG("failed to save net with fd %d ret %d errno %d", fd, ret, errno);
  return -1;
}

RecurNN*
rnn_load_net(const char *filename){
  int fd;
  uint vlen;
  int ret = 0;
  RecurNN tmpnet;
  RecurNNBPTT tmpbptt;
  RecurNN *net;

  fd = open(filename, O_RDONLY);

#define READ_SCALAR(obj, attr) do { ret = cdb_seek(fd, QUOTE(attr),     \
        strlen(QUOTE(attr)), &vlen);                                    \
    if (ret < 1){ DEBUG("error %d loading '%s'", ret, QUOTE(attr));     \
      goto pre_alloc_error;}                                                      \
    if (vlen != sizeof(obj.attr)) {                                    \
      DEBUG("size mismatch on '%s->%s' want %u, found %lu",              \
          QUOTE(obj), QUOTE(attr), vlen, sizeof(obj.attr));            \
      goto pre_alloc_error;                                             \
    }                                                                   \
    cdb_bread(fd, &obj.attr, vlen);                                    \
  } while (0)

  READ_SCALAR(tmpnet, i_size);
  READ_SCALAR(tmpnet, h_size);
  READ_SCALAR(tmpnet, o_size);

  READ_SCALAR(tmpnet, input_size);
  READ_SCALAR(tmpnet, hidden_size);
  READ_SCALAR(tmpnet, output_size);

  READ_SCALAR(tmpnet, ih_size);
  READ_SCALAR(tmpnet, ho_size);
  READ_SCALAR(tmpnet, bias);
  READ_SCALAR(tmpnet, rng);
  READ_SCALAR(tmpnet, generation);
  READ_SCALAR(tmpnet, flags);
  READ_SCALAR(tmpbptt, depth);
  READ_SCALAR(tmpbptt, batch_size);
  READ_SCALAR(tmpbptt, learn_rate);
  READ_SCALAR(tmpbptt, index);
  READ_SCALAR(tmpbptt, momentum);
  READ_SCALAR(tmpbptt, momentum_weight);

#undef READ_SCALAR

  net = rnn_new(tmpnet.input_size, tmpnet.hidden_size,
      tmpnet.output_size, tmpnet.flags, 0, NULL,
      tmpbptt.depth, tmpbptt.learn_rate, tmpbptt.momentum,
      tmpbptt.momentum_weight, tmpbptt.batch_size);
  net->rng = tmpnet.rng;
  net->generation = tmpnet.generation;
  net->bptt->index = tmpbptt.index;

#define CHECK_SCALAR(new, tmp, attr) do {                                   \
  if (new->attr != tmp.attr){                                              \
    DEBUG("attribute '%s' differs %x vs %x", QUOTE(attr),               \
        (int)new->attr, (int)tmp.attr);                                \
    goto error;}                                                        \
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
  CHECK_SCALAR(net->bptt, tmpbptt, depth);
  CHECK_SCALAR(net->bptt, tmpbptt, batch_size);
  CHECK_SCALAR(net->bptt, tmpbptt, index);
#undef CHECK_SCALAR

  /* so presumably all the arrays are all allocate and the right size. */
#define READ_ARRAY(obj, attr, size) do { ret = cdb_seek(fd, QUOTE(attr), \
        strlen(QUOTE(attr)), &vlen);                                    \
    if (ret < 1){ DEBUG("error %d loading '%s'", ret, QUOTE(attr));     \
      goto error;}                                                      \
    if (vlen != size) {                                                 \
      DEBUG("array size mismatch on '%s->%s' saved %u, calculated %lu",  \
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

  READ_ARRAY(net->bptt, i_error, net->i_size * sizeof(float));
  READ_ARRAY(net->bptt, h_error, net->h_size * sizeof(float));
  READ_ARRAY(net->bptt, o_error, net->o_size * sizeof(float));
  READ_ARRAY(net->bptt, ih_momentum, net->ih_size * sizeof(float));
  READ_ARRAY(net->bptt, ho_momentum, net->ho_size * sizeof(float));
  READ_ARRAY(net->bptt, history, net->bptt->depth * net->i_size * sizeof(float));
  READ_ARRAY(net->bptt, ih_delta, net->ih_size * sizeof(float));

#undef READ_ARRAY
  close(fd);
  DEBUG("successfully loaded net '%s'", filename);
  return net;
 error:
  rnn_delete_net(net);
 pre_alloc_error:
  DEBUG("loading net failed!");
  close(fd);
  return NULL;
}
