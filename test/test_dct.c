#include "mfcc.h"
#include "pgm_dump.h"

#define CYCLES 1000000

int
main(void){
#define len 13
  float input[len] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 4};
  float output[len];
  float recycle[len];
  float sum;
  START_TIMER(naive);
  sum = 0;
  for (int i = 0; i < CYCLES; i++){
    recur_dct(input, output, len);
    sum += output[3];
  }
  DEBUG("sum %g", sum);
  DEBUG_TIMER(naive);
  START_TIMER(cache);
  sum = 0;
  for (int i = 0; i < CYCLES; i++){
    recur_dct_cached(input, output, len);
    sum += output[3];
  }
  DEBUG("sum %g", sum);
  DEBUG_TIMER(cache);

  for (int i = 0; i < 10; i++){
    recur_dct(input, output, len);
    //recur_dct_cached(input, output, len);
    recur_idct(output, recycle, len);
  }
  for (int i = 0; i < len; i++){
    printf("in %6.6f  out  %6.6f  rec  %6.6f\n",
        input[i], output[i], recycle[i]);
  }
}
