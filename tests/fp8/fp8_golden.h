#ifndef FP8_GOLDEN_H
#define FP8_GOLDEN_H

#include <stdint.h>

/*----------------------------------------------------------------------------
| Wire format shared with gen_golden.py.  Little-endian and fixed-width; only
| raw bit patterns cross this boundary, never decimal text.
*----------------------------------------------------------------------------*/
#define FP8_GOLDEN_MAGIC      "FP8GOLD\x01"
#define FP8_GOLDEN_MAGIC_LEN  8
#define FP8_GOLDEN_NAME_LEN   32
#define FP8_GOLDEN_SECHDR_LEN 48
#define FP8_GOLDEN_REC_LEN    20

/*----------------------------------------------------------------------------
| File   := magic[8] nsections:u32 section*
| Section:= name[32] op:u32 fmt:u32 count:u32 reclen:u32 record[count]
| Record := in0:u32 in1:u32 in2:u32 out:u32 flags:u8 flagsKnown:u8 rmode:u8
|           rsvd:u8
| 'rmode' is a softfloat_roundingMode value; 0 (near_even) for torch goldens.
*----------------------------------------------------------------------------*/

enum fp8_golden_op {
    FP8_OP_WIDEN_F32 = 1,
    FP8_OP_ADD       = 2,
    FP8_OP_SUB       = 3,
    FP8_OP_MUL       = 4,
    FP8_OP_DIV       = 5,
    FP8_OP_EQ        = 6,
    FP8_OP_LE        = 7,
    FP8_OP_LT        = 8,
    FP8_OP_NARROW_F8 = 9
};

/*----------------------------------------------------------------------------
| 'fmt' holds the softfloat_fp8Format value verbatim, so no mapping is needed.
*----------------------------------------------------------------------------*/

#endif
