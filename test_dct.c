#include "recur-context.h"
#include "pgm_dump.h"


int
main(void){
#define len 10
  float input[len] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 4};
  float output[len];
  float recycle[len];
  recur_dct(input, output, len);
  recur_idct(output, recycle, len);

  for (int i = 0; i < len; i++){
    printf("in %6.6f  out  %6.6f  rec  %6.6f\n",
        input[i], output[i], recycle[i]);
  }
}
