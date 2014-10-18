#include "recur-nn.h"
#include "recur-nn-helpers.h"
#include <stdio.h>

#define IN_RANGE_01(x) (((x) >= 0.0f) && ((x) <= 1.0f))

/*restrict to 0-1 range (mostly for probabilities)*/
static UNUSED char *
opt_set_floatval01(const char *arg, float *f){
  char *msg = opt_set_floatval(arg, f);
  if (msg == NULL && ! IN_RANGE_01(*f)){
    char *s;
    if (asprintf(&s, "We want a number between 0 and 1, not '%s'", arg) > 0){
      return s;
    }
  }
  return msg;
}
