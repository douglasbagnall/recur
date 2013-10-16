/* Copyright (c) 2013 Douglas Bagnall <douglas@halo.gen.nz> */
#ifndef __RECUR_COMMON_H__
#define __RECUR_COMMON_H__

#include "recur-config.h"
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#ifndef UNUSED
#define UNUSED __attribute__ ((unused))
#else
#warning UNUSED is set
#endif

#ifndef INVISIBLE
#define INVISIBLE __attribute__ ((visibility("hidden")))
#else
#warning INVISIBLE is set
#endif

#define VERBOSE_DEBUG 0

#define streq(a,b) (strcmp((a),(b)) == 0)

#define QUOTE_(x) #x
#define QUOTE(x) QUOTE_(x)

#define STDERR_DEBUG(format, ...) do {                                 \
    fprintf (stderr, (format),## __VA_ARGS__);                  \
    fputc('\n', stderr);                                         \
    fflush(stderr);                                             \
    } while (0)

#ifdef GST_DEBUG
#define DEBUG_LINENO() GST_DEBUG("%-25s  line %4d ooo\n", __func__, __LINE__ )
#define DEBUG(args...) GST_DEBUG(args)
#define MAYBE_DEBUG(args...) GST_LOG(args)
#else
#define DEBUG STDERR_DEBUG
#if VERBOSE_DEBUG
#define MAYBE_DEBUG(args...) STDERR_DEBUG(args)
#else
#define MAYBE_DEBUG(args...) /*args */
#endif

#define DEBUG_LINENO() do {                                             \
        DEBUG("%s:%d:0: note: in %s, line %d\n",                        \
              __FILE__, __LINE__, __func__, __LINE__);                  \
        fflush(stderr);}                                                \
    while(0)
#endif

#define ROUND_UP_16(x)  (((x) + 15) & ~15UL)
#define ROUND_UP_4(x)  (((x) + 3) & ~3UL)

#define RECUR_LOG_COLOUR GST_DEBUG_FG_GREEN | GST_DEBUG_BOLD

/* short type names */
typedef uint64_t u64;
typedef int64_t s64;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint16_t u16;
typedef int16_t s16;
typedef uint8_t u8;
typedef int8_t s8;
typedef unsigned int uint;

/*memory allocation */
#define ALIGNMENT 16
static inline __attribute__((malloc)) void *
malloc_aligned_or_die(size_t size){
  void *mem;
  int err = posix_memalign(&mem, ALIGNMENT, size);
  if (err){
    fprintf(stderr, "posix_memalign returned %d trying to allocate "
        "%zu bytes aligned on %u byte boundaries\n", err, size, ALIGNMENT);
    abort();
  }
  return mem;
}

static inline __attribute__((malloc)) void *
zalloc_aligned_or_die(size_t size){
  void *mem = malloc_aligned_or_die(size);
  memset(mem, 0, size);
  return mem;
}

#define BILLION 1000000000L

#ifndef __clang__
#define ASSUME_ALIGNED(x)   (x) = __builtin_assume_aligned ((x), 16)
#else
#define ASSUME_ALIGNED(x) /* x */
#endif

static inline void
recur_start_timer(struct timespec *time)
{
  clock_gettime(CLOCK_MONOTONIC, time);
}

static inline s64
recur_read_timer(struct timespec *start_time)
{
  struct timespec end_time;
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  s64 secs = end_time.tv_sec - start_time->tv_sec;
  s64 nano = end_time.tv_nsec - start_time->tv_nsec;
  return secs * BILLION + nano;
}

#define START_TIMER(x) struct timespec timer___ ## x; recur_start_timer(&timer___ ## x);
#define READ_TIMER(x) recur_read_timer(&timer___ ## x)
#define DEBUG_TIMER(x) do {s64 res___ ## x = recur_read_timer(&timer___ ## x); \
    DEBUG(QUOTE(x) " took %.3g seconds (%.3g microseconds, %.3g frame)", \
        res___ ## x * (1.0 / BILLION), res___ ## x * (1000000.0 / BILLION), res___ ## x * (25.0 / BILLION)); \
  } while (0)

#ifndef MAX
#define MAX(a, b)  (((a) >= (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b)  (((a) < (b)) ? (a) : (b))
#endif

static void __attribute__((noinline)) UNUSED
BP(void) { asm (""); }

#define CBP(x) do { if (x) BP(); } while(0)

#endif
