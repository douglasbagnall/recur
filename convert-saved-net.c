#include "recur-nn.h"

int
main(int argc, char *argv[])
{
  RecurNN *net = rnn_load_net(argv[1]);
  net->flags &= ~RNN_NET_FLAG_OWN_BPTT;
  rnn_save_net(net, argv[2], 1);
}
