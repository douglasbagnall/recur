#include "mfcc.h"
#include "pgm_dump.h"



typedef struct {
  int window;
  char *name;
} window_lut;

#define HEIGHT 400
#define WIDTH 300


int
main(void){
  uint i;
  window_lut windows[] = {
    {RECUR_WINDOW_NONE, "none"},
    {RECUR_WINDOW_HANN, "hann"},
    {RECUR_WINDOW_VORBIS, "vorbis"},
    {RECUR_WINDOW_MP3, "mp3"}
  };
  float *mask = malloc(HEIGHT * sizeof(float));
  u8 *pgm = malloc(HEIGHT * WIDTH);
  char name[200];

  for (i = 0; i < sizeof(windows) / sizeof(windows[0]); i++){
    window_lut *w = &windows[i];
    recur_window_init(mask, HEIGHT, w->window, WIDTH - 1);
    memset(pgm, 0, HEIGHT * WIDTH);
    for (int y = 0; y < HEIGHT; y++){
      for (int x = 0; x < mask[y]; x++){
        pgm[y * WIDTH + x] = 100;
      }
      pgm[y * WIDTH + (int)mask[y]] = 255;
    }
    snprintf(name, sizeof(name), "%s/window-%s.pgm", IMAGE_DIR, w->name);

    pgm_dump(pgm, WIDTH, HEIGHT, name);
  }
}
