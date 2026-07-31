/* Wire format shared by the spike target and the host checkers.
   A section is a header plus `count` packed records; a record is the inputs
   the target really used, the result it produced, and its fflags byte. */
#ifndef FP8_VECTOR_STREAM_H
#define FP8_VECTOR_STREAM_H

#include <stdint.h>

#define FP8VEC_MAGIC "FP8VEC\x01\x00"
#define FP8VEC_MAGIC_LEN 8
#define FP8VEC_END "ENDFP8VEC\x00\x00\x00"
#define FP8VEC_END_LEN 12
#define FP8VEC_NAME_LEN 32

/* poison written into every result slot before the instruction runs */
#define FP8VEC_POISON8  0xA5
#define FP8VEC_POISON16 0xA5A5

typedef struct {
  char name[FP8VEC_NAME_LEN];
  uint8_t n_in;      /* 1..3 */
  uint8_t in_w[3];   /* bytes per input, little endian in the record */
  uint8_t out_w;     /* bytes of result */
  uint8_t frm;       /* rounding mode in force */
  uint8_t rsvd[2];
  uint32_t count;
} fp8vec_hdr;

#endif
