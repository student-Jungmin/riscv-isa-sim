/* Runs every fp8 (SEW=8) vector instruction on spike and streams what it did.
   Nothing is judged here: inputs, results and fflags all go to stdout so the
   host checkers decide.  pk cannot create files, so stdout is the channel. */

#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include "stream.h"

/* ------------------------------------------------------------------ output */

static uint8_t obuf[1 << 16];
static uint32_t olen;

static void oflush(void)
{
  uint32_t off = 0;
  while (off < olen) {
    long n = write(1, obuf + off, olen - off);
    if (n <= 0) break;
    off += (uint32_t)n;
  }
  olen = 0;
}

static void oput(const void *p, uint32_t n)
{
  if (olen + n > sizeof obuf) oflush();
  memcpy(obuf + olen, p, n);
  olen += n;
}

static void sec_begin(const char *name, uint8_t n_in, uint8_t w0, uint8_t w1,
                      uint8_t w2, uint8_t out_w, uint8_t frm, uint32_t count)
{
  fp8vec_hdr h;
  size_t n = strlen(name);
  memset(&h, 0, sizeof h);
  if (n > FP8VEC_NAME_LEN - 1) n = FP8VEC_NAME_LEN - 1;
  memcpy(h.name, name, n);
  h.n_in = n_in;
  h.in_w[0] = w0; h.in_w[1] = w1; h.in_w[2] = w2;
  h.out_w = out_w;
  h.frm = frm;
  h.count = count;
  oput(&h, sizeof h);
}

static void rec1(uint64_t a, uint64_t o, uint8_t f, int aw, int ow)
{
  oput(&a, aw); oput(&o, ow); oput(&f, 1);
}
static void rec2(uint64_t a, uint64_t b, uint64_t o, uint8_t f,
                 int aw, int bw, int ow)
{
  oput(&a, aw); oput(&b, bw); oput(&o, ow); oput(&f, 1);
}
static void rec3(uint64_t a, uint64_t b, uint64_t c, uint64_t o, uint8_t f,
                 int aw, int bw, int cw, int ow)
{
  oput(&a, aw); oput(&b, bw); oput(&c, cw); oput(&o, ow); oput(&f, 1);
}

/* --------------------------------------------------------------- utilities */

/* An fp8 scalar operand must be NaN-boxed: bits 63:8 all ones. */
static double bx8(uint8_t v)
{
  uint64_t u = 0xFFFFFFFFFFFFFF00ULL | v;
  double d;
  memcpy(&d, &u, 8);
  return d;
}

static void set_frm(unsigned m)
{
  asm volatile("csrw frm, %0" :: "r"((unsigned long)m));
}

static uint64_t rng_s;
static uint64_t rng(void)
{
  rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17;
  return rng_s;
}

/* Structured fp8 probes: zeros, both subnormal ends, min normal, ties,
   powers, max finite, both NaN encodings, and points either side of them. */
#define NT8 48
static const uint8_t T8[NT8] = {
  0x00, 0x80, 0x01, 0x81, 0x02, 0x03, 0x04, 0x07,
  0x87, 0x08, 0x88, 0x09, 0x0F, 0x10, 0x18, 0x1F,
  0x20, 0x28, 0x2F, 0x30, 0xB0, 0x34, 0x37, 0x38,
  0xB8, 0x39, 0x3B, 0x3F, 0x40, 0xC0, 0x44, 0x48,
  0xC8, 0x4F, 0x50, 0x58, 0x60, 0x68, 0x6F, 0x70,
  0x77, 0x78, 0x7B, 0x7D, 0x7E, 0xFE, 0x7F, 0xFF,
};

/* Structured f16 probes for the widening arms: fp8-representable values,
   values between fp8 neighbours, the fp8 overflow edge, inf and both NaNs. */
#define NT16 48
static const uint16_t T16[NT16] = {
  0x0000, 0x8000, 0x0001, 0x8001, 0x0200, 0x0400, 0x0800, 0x1000,
  0x1400, 0x9400, 0x1800, 0x1C00, 0x2000, 0xA000, 0x2400, 0x2800,
  0x3000, 0xB000, 0x3400, 0x3800, 0xB800, 0x3A00, 0x3C00, 0xBC00,
  0x3E00, 0x4000, 0xC000, 0x4200, 0x4400, 0x4800, 0x4C00, 0x5000,
  0x5400, 0x5800, 0x5C00, 0x5F00, 0x5F80, 0xDF80, 0x5FC0, 0x6000,
  0x7000, 0x7800, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01,
};

/* --------------------------------------------------- instruction wrappers */

#define ASM_HEAD "csrw fflags, zero\n\t vsetivli t0, 1, e8, m1, ta, ma\n\t"
#define ASM_TAIL "csrr %0, fflags"

/* vd, vs2, vs1 with fp8 in and out */
#define DEF_VV(fn, MNEM) \
static uint8_t fn(uint8_t a, uint8_t b, uint8_t *fl) { \
  uint8_t A = a, B = b, O = FP8VEC_POISON8; unsigned long f; \
  asm volatile(ASM_HEAD \
    "vle8.v v8, (%1)\n\t vle8.v v9, (%2)\n\t" \
    MNEM " v10, v8, v9\n\t vse8.v v10, (%3)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&B), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

/* vd, vs2, rs1 with fp8 in and out */
#define DEF_VF(fn, MNEM) \
static uint8_t fn(uint8_t a, uint8_t b, uint8_t *fl) { \
  uint8_t A = a, O = FP8VEC_POISON8; double S = bx8(b); unsigned long f; \
  asm volatile(ASM_HEAD \
    "vle8.v v8, (%1)\n\t" MNEM " v10, v8, %2\n\t vse8.v v10, (%3)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "f"(S), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

/* mask-producing compare; vsm.v spills the mask and bit 0 is the answer.
   vmv.x.s is not usable here: this fork asserts on it. */
#define DEF_CMP_VV(fn, MNEM) \
static uint8_t fn(uint8_t a, uint8_t b, uint8_t *fl) { \
  uint8_t A = a, B = b, M = FP8VEC_POISON8; unsigned long f; \
  asm volatile(ASM_HEAD \
    "vle8.v v8, (%1)\n\t vle8.v v9, (%2)\n\t vmv.v.i v10, 0\n\t" \
    MNEM " v10, v8, v9\n\t vsm.v v10, (%3)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&B), "r"(&M) : "memory", "t0"); \
  *fl = (uint8_t)f; return (uint8_t)(M & 1); }

#define DEF_CMP_VF(fn, MNEM) \
static uint8_t fn(uint8_t a, uint8_t b, uint8_t *fl) { \
  uint8_t A = a, M = FP8VEC_POISON8; double S = bx8(b); unsigned long f; \
  asm volatile(ASM_HEAD \
    "vle8.v v8, (%1)\n\t vmv.v.i v10, 0\n\t" \
    MNEM " v10, v8, %2\n\t vsm.v v10, (%3)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "f"(S), "r"(&M) : "memory", "t0"); \
  *fl = (uint8_t)f; return (uint8_t)(M & 1); }

/* Fused ops read vd, so the destination is preloaded with c.  The .vv fused
   forms name vs1 before vs2 in assembly, unlike vfadd.vv and friends, so v8
   holds vs2 and appears second. */
#define DEF_FMA_VV(fn, MNEM) \
static uint8_t fn(uint8_t a, uint8_t b, uint8_t c, uint8_t *fl) { \
  uint8_t A = a, B = b, C = c, O = FP8VEC_POISON8; unsigned long f; \
  asm volatile(ASM_HEAD \
    "vle8.v v8, (%1)\n\t vle8.v v9, (%2)\n\t vle8.v v10, (%3)\n\t" \
    MNEM " v10, v9, v8\n\t vse8.v v10, (%4)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&B), "r"(&C), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

#define DEF_FMA_VF(fn, MNEM) \
static uint8_t fn(uint8_t a, uint8_t b, uint8_t c, uint8_t *fl) { \
  uint8_t A = a, C = c, O = FP8VEC_POISON8; double S = bx8(b); unsigned long f; \
  asm volatile(ASM_HEAD \
    "vle8.v v8, (%1)\n\t vle8.v v10, (%2)\n\t" \
    MNEM " v10, %3, v8\n\t vse8.v v10, (%4)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&C), "f"(S), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

/* fp8 in, fp8 out */
#define DEF_V(fn, MNEM) \
static uint8_t fn(uint8_t a, uint8_t *fl) { \
  uint8_t A = a, O = FP8VEC_POISON8; unsigned long f; \
  asm volatile(ASM_HEAD \
    "vle8.v v8, (%1)\n\t" MNEM " v10, v8\n\t vse8.v v10, (%2)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

/* SEW=8 in, SEW=8 out, but the element is an integer on one side */
#define DEF_V_RAW(fn, MNEM) DEF_V(fn, MNEM)

/* widening vv: fp8 sources, f16 destination at LMUL*2 */
#define DEF_W_VV(fn, MNEM) \
static uint16_t fn(uint8_t a, uint8_t b, uint8_t *fl) { \
  uint8_t A = a, B = b; uint16_t O = FP8VEC_POISON16; unsigned long f; \
  asm volatile(ASM_HEAD \
    "vle8.v v8, (%1)\n\t vle8.v v9, (%2)\n\t" \
    MNEM " v16, v8, v9\n\t vsetivli t0, 1, e16, m2, ta, ma\n\t" \
    "vse16.v v16, (%3)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&B), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

#define DEF_W_VF(fn, MNEM) \
static uint16_t fn(uint8_t a, uint8_t b, uint8_t *fl) { \
  uint8_t A = a; uint16_t O = FP8VEC_POISON16; double S = bx8(b); unsigned long f; \
  asm volatile(ASM_HEAD \
    "vle8.v v8, (%1)\n\t" MNEM " v16, v8, %2\n\t" \
    "vsetivli t0, 1, e16, m2, ta, ma\n\t vse16.v v16, (%3)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "f"(S), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

/* .wv / .wf: vs2 is already f16, vs1 or rs1 is fp8 */
#define DEF_W_WV(fn, MNEM) \
static uint16_t fn(uint16_t a, uint8_t b, uint8_t *fl) { \
  uint16_t A = a, O = FP8VEC_POISON16; uint8_t B = b; unsigned long f; \
  asm volatile("csrw fflags, zero\n\t vsetivli t0, 1, e16, m2, ta, ma\n\t" \
    "vle16.v v16, (%1)\n\t vsetivli t0, 1, e8, m1, ta, ma\n\t" \
    "vle8.v v9, (%2)\n\t" MNEM " v20, v16, v9\n\t" \
    "vsetivli t0, 1, e16, m2, ta, ma\n\t vse16.v v20, (%3)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&B), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

#define DEF_W_WF(fn, MNEM) \
static uint16_t fn(uint16_t a, uint8_t b, uint8_t *fl) { \
  uint16_t A = a, O = FP8VEC_POISON16; double S = bx8(b); unsigned long f; \
  asm volatile("csrw fflags, zero\n\t vsetivli t0, 1, e16, m2, ta, ma\n\t" \
    "vle16.v v16, (%1)\n\t vsetivli t0, 1, e8, m1, ta, ma\n\t" \
    MNEM " v20, v16, %2\n\t" \
    "vsetivli t0, 1, e16, m2, ta, ma\n\t vse16.v v20, (%3)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "f"(S), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

/* widening fused: fp8 x fp8 accumulated into the f16 destination */
#define DEF_W_FMA_VV(fn, MNEM) \
static uint16_t fn(uint8_t a, uint8_t b, uint16_t c, uint8_t *fl) { \
  uint8_t A = a, B = b; uint16_t C = c, O = FP8VEC_POISON16; unsigned long f; \
  asm volatile("csrw fflags, zero\n\t vsetivli t0, 1, e16, m2, ta, ma\n\t" \
    "vle16.v v16, (%3)\n\t vsetivli t0, 1, e8, m1, ta, ma\n\t" \
    "vle8.v v8, (%1)\n\t vle8.v v9, (%2)\n\t" \
    MNEM " v16, v9, v8\n\t vsetivli t0, 1, e16, m2, ta, ma\n\t" \
    "vse16.v v16, (%4)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&B), "r"(&C), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

#define DEF_W_FMA_VF(fn, MNEM) \
static uint16_t fn(uint8_t a, uint8_t b, uint16_t c, uint8_t *fl) { \
  uint8_t A = a; uint16_t C = c, O = FP8VEC_POISON16; double S = bx8(b); \
  unsigned long f; \
  asm volatile("csrw fflags, zero\n\t vsetivli t0, 1, e16, m2, ta, ma\n\t" \
    "vle16.v v16, (%2)\n\t vsetivli t0, 1, e8, m1, ta, ma\n\t" \
    "vle8.v v8, (%1)\n\t" MNEM " v16, %3, v8\n\t" \
    "vsetivli t0, 1, e16, m2, ta, ma\n\t vse16.v v16, (%4)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&C), "f"(S), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

/* SEW=8 source, 16-bit destination (vfwcvt integer forms) */
#define DEF_W_CVT(fn, MNEM) \
static uint16_t fn(uint8_t a, uint8_t *fl) { \
  uint8_t A = a; uint16_t O = FP8VEC_POISON16; unsigned long f; \
  asm volatile(ASM_HEAD \
    "vle8.v v8, (%1)\n\t" MNEM " v16, v8\n\t" \
    "vsetivli t0, 1, e16, m2, ta, ma\n\t vse16.v v16, (%2)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

/* 16-bit source, SEW=8 destination (vfncvt forms) */
#define DEF_N_CVT(fn, MNEM) \
static uint8_t fn(uint16_t a, uint8_t *fl) { \
  uint16_t A = a; uint8_t O = FP8VEC_POISON8; unsigned long f; \
  asm volatile("csrw fflags, zero\n\t vsetivli t0, 1, e16, m2, ta, ma\n\t" \
    "vle16.v v16, (%1)\n\t vsetivli t0, 1, e8, m1, ta, ma\n\t" \
    MNEM " v8, v16\n\t vse8.v v8, (%2)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

/* Negative control: -DFP8V_SABOTAGE hands vfsub.vv its operands the wrong way
   round.  A harness that cannot see this cannot see a real wiring bug. */
#ifdef FP8V_SABOTAGE
#define DEF_VV_SUB(fn, MNEM) \
static uint8_t fn(uint8_t a, uint8_t b, uint8_t *fl) { \
  uint8_t A = a, B = b, O = FP8VEC_POISON8; unsigned long f; \
  asm volatile(ASM_HEAD \
    "vle8.v v8, (%1)\n\t vle8.v v9, (%2)\n\t" \
    MNEM " v10, v9, v8\n\t vse8.v v10, (%3)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&A), "r"(&B), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }
#else
#define DEF_VV_SUB DEF_VV
#endif

DEF_VV(i_vfadd_vv,   "vfadd.vv")
DEF_VV_SUB(i_vfsub_vv, "vfsub.vv")
DEF_VV(i_vfmul_vv,   "vfmul.vv")
DEF_VV(i_vfdiv_vv,   "vfdiv.vv")
DEF_VV(i_vfmin_vv,   "vfmin.vv")
DEF_VV(i_vfmax_vv,   "vfmax.vv")
DEF_VV(i_vfsgnj_vv,  "vfsgnj.vv")
DEF_VV(i_vfsgnjn_vv, "vfsgnjn.vv")
DEF_VV(i_vfsgnjx_vv, "vfsgnjx.vv")

DEF_VF(i_vfadd_vf,   "vfadd.vf")
DEF_VF(i_vfsub_vf,   "vfsub.vf")
DEF_VF(i_vfrsub_vf,  "vfrsub.vf")
DEF_VF(i_vfmul_vf,   "vfmul.vf")
DEF_VF(i_vfdiv_vf,   "vfdiv.vf")
DEF_VF(i_vfrdiv_vf,  "vfrdiv.vf")
DEF_VF(i_vfmin_vf,   "vfmin.vf")
DEF_VF(i_vfmax_vf,   "vfmax.vf")
DEF_VF(i_vfsgnj_vf,  "vfsgnj.vf")
DEF_VF(i_vfsgnjn_vf, "vfsgnjn.vf")
DEF_VF(i_vfsgnjx_vf, "vfsgnjx.vf")

DEF_CMP_VV(i_vmfeq_vv, "vmfeq.vv")
DEF_CMP_VV(i_vmfne_vv, "vmfne.vv")
DEF_CMP_VV(i_vmflt_vv, "vmflt.vv")
DEF_CMP_VV(i_vmfle_vv, "vmfle.vv")
DEF_CMP_VF(i_vmfeq_vf, "vmfeq.vf")
DEF_CMP_VF(i_vmfne_vf, "vmfne.vf")
DEF_CMP_VF(i_vmflt_vf, "vmflt.vf")
DEF_CMP_VF(i_vmfle_vf, "vmfle.vf")
DEF_CMP_VF(i_vmfgt_vf, "vmfgt.vf")
DEF_CMP_VF(i_vmfge_vf, "vmfge.vf")

DEF_FMA_VV(i_vfmacc_vv,  "vfmacc.vv")
DEF_FMA_VV(i_vfnmacc_vv, "vfnmacc.vv")
DEF_FMA_VV(i_vfmsac_vv,  "vfmsac.vv")
DEF_FMA_VV(i_vfnmsac_vv, "vfnmsac.vv")
DEF_FMA_VV(i_vfmadd_vv,  "vfmadd.vv")
DEF_FMA_VV(i_vfnmadd_vv, "vfnmadd.vv")
DEF_FMA_VV(i_vfmsub_vv,  "vfmsub.vv")
DEF_FMA_VV(i_vfnmsub_vv, "vfnmsub.vv")
DEF_FMA_VF(i_vfmacc_vf,  "vfmacc.vf")
DEF_FMA_VF(i_vfnmacc_vf, "vfnmacc.vf")
DEF_FMA_VF(i_vfmsac_vf,  "vfmsac.vf")
DEF_FMA_VF(i_vfnmsac_vf, "vfnmsac.vf")
DEF_FMA_VF(i_vfmadd_vf,  "vfmadd.vf")
DEF_FMA_VF(i_vfnmadd_vf, "vfnmadd.vf")
DEF_FMA_VF(i_vfmsub_vf,  "vfmsub.vf")
DEF_FMA_VF(i_vfnmsub_vf, "vfnmsub.vf")

DEF_V(i_vfsqrt_v,   "vfsqrt.v")
DEF_V(i_vfrec7_v,   "vfrec7.v")
DEF_V(i_vfrsqrt7_v, "vfrsqrt7.v")

DEF_V_RAW(i_vfcvt_x_f_v,      "vfcvt.x.f.v")
DEF_V_RAW(i_vfcvt_xu_f_v,     "vfcvt.xu.f.v")
DEF_V_RAW(i_vfcvt_rtz_x_f_v,  "vfcvt.rtz.x.f.v")
DEF_V_RAW(i_vfcvt_rtz_xu_f_v, "vfcvt.rtz.xu.f.v")
DEF_V_RAW(i_vfcvt_f_x_v,      "vfcvt.f.x.v")
DEF_V_RAW(i_vfcvt_f_xu_v,     "vfcvt.f.xu.v")

DEF_W_VV(i_vfwadd_vv, "vfwadd.vv")
DEF_W_VV(i_vfwsub_vv, "vfwsub.vv")
DEF_W_VV(i_vfwmul_vv, "vfwmul.vv")
DEF_W_VF(i_vfwadd_vf, "vfwadd.vf")
DEF_W_VF(i_vfwsub_vf, "vfwsub.vf")
DEF_W_VF(i_vfwmul_vf, "vfwmul.vf")
DEF_W_WV(i_vfwadd_wv, "vfwadd.wv")
DEF_W_WV(i_vfwsub_wv, "vfwsub.wv")
DEF_W_WF(i_vfwadd_wf, "vfwadd.wf")
DEF_W_WF(i_vfwsub_wf, "vfwsub.wf")

DEF_W_FMA_VV(i_vfwmacc_vv,  "vfwmacc.vv")
DEF_W_FMA_VV(i_vfwnmacc_vv, "vfwnmacc.vv")
DEF_W_FMA_VV(i_vfwmsac_vv,  "vfwmsac.vv")
DEF_W_FMA_VV(i_vfwnmsac_vv, "vfwnmsac.vv")
DEF_W_FMA_VF(i_vfwmacc_vf,  "vfwmacc.vf")
DEF_W_FMA_VF(i_vfwnmacc_vf, "vfwnmacc.vf")
DEF_W_FMA_VF(i_vfwmsac_vf,  "vfwmsac.vf")
DEF_W_FMA_VF(i_vfwnmsac_vf, "vfwnmsac.vf")

DEF_W_CVT(i_vfwcvt_f_f_v,      "vfwcvt.f.f.v")
DEF_W_CVT(i_vfwcvt_x_f_v,      "vfwcvt.x.f.v")
DEF_W_CVT(i_vfwcvt_xu_f_v,     "vfwcvt.xu.f.v")
DEF_W_CVT(i_vfwcvt_rtz_x_f_v,  "vfwcvt.rtz.x.f.v")
DEF_W_CVT(i_vfwcvt_rtz_xu_f_v, "vfwcvt.rtz.xu.f.v")

DEF_N_CVT(i_vfncvt_f_f_w,     "vfncvt.f.f.w")
DEF_N_CVT(i_vfncvt_rod_f_f_w, "vfncvt.rod.f.f.w")
DEF_N_CVT(i_vfncvt_f_x_w,     "vfncvt.f.x.w")
DEF_N_CVT(i_vfncvt_f_xu_w,    "vfncvt.f.xu.w")

/* ------------------------------------------------------- reductions, moves */

#define RED_VL 8

#define DEF_RED(fn, MNEM) \
static uint8_t fn(const uint8_t *v, uint8_t s, uint8_t *fl) { \
  uint8_t S = s, O = FP8VEC_POISON8; unsigned long f; \
  asm volatile("csrw fflags, zero\n\t vsetivli t0, 8, e8, m1, ta, ma\n\t" \
    "vle8.v v8, (%1)\n\t vsetivli t0, 1, e8, m1, ta, ma\n\t" \
    "vle8.v v12, (%2)\n\t vsetivli t0, 8, e8, m1, ta, ma\n\t" \
    MNEM " v10, v8, v12\n\t vsetivli t0, 1, e8, m1, ta, ma\n\t" \
    "vse8.v v10, (%3)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(v), "r"(&S), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

#define DEF_WRED(fn, MNEM) \
static uint16_t fn(const uint8_t *v, uint16_t s, uint8_t *fl) { \
  uint16_t S = s, O = FP8VEC_POISON16; unsigned long f; \
  asm volatile("csrw fflags, zero\n\t vsetivli t0, 1, e16, m1, ta, ma\n\t" \
    "vle16.v v12, (%2)\n\t vsetivli t0, 8, e8, m1, ta, ma\n\t" \
    "vle8.v v8, (%1)\n\t" MNEM " v10, v8, v12\n\t" \
    "vsetivli t0, 1, e16, m1, ta, ma\n\t vse16.v v10, (%3)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(v), "r"(&S), "r"(&O) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

DEF_RED(i_vfredosum_vs, "vfredosum.vs")
DEF_RED(i_vfredusum_vs, "vfredusum.vs")
DEF_RED(i_vfredmin_vs,  "vfredmin.vs")
DEF_RED(i_vfredmax_vs,  "vfredmax.vs")
DEF_WRED(i_vfwredosum_vs, "vfwredosum.vs")
DEF_WRED(i_vfwredusum_vs, "vfwredusum.vs")

/* vfmv.s.f then vfmv.f.s: the pair must round-trip every fp8 encoding */
static uint64_t i_vfmv_roundtrip(uint8_t a, uint8_t *out_vec)
{
  double S = bx8(a), R;
  uint8_t O = FP8VEC_POISON8;
  uint64_t u;
  asm volatile("vsetivli t0, 4, e8, m1, ta, ma\n\t"
    "vmv.v.i v10, 0\n\t vfmv.s.f v10, %1\n\t vse8.v v10, (%2)\n\t"
    "vfmv.f.s %0, v10"
    : "=f"(R) : "f"(S), "r"(&O) : "memory", "t0");
  memcpy(&u, &R, 8);
  *out_vec = O;
  return u;
}

static uint8_t i_vfmv_v_f(uint8_t a)
{
  double S = bx8(a);
  uint8_t O = FP8VEC_POISON8;
  asm volatile("vsetivli t0, 4, e8, m1, ta, ma\n\t"
    "vfmv.v.f v10, %0\n\t vse8.v v10, (%1)"
    :: "f"(S), "r"(&O) : "memory", "t0");
  return O;
}

/* mask bit 0 selects the scalar, bit 1 keeps vs2 */
static uint16_t i_vfmerge_vfm(uint8_t a, uint8_t b)
{
  uint8_t V[2] = { a, a }, O[2] = { FP8VEC_POISON8, FP8VEC_POISON8 };
  unsigned long m = 1;
  double S = bx8(b);
  asm volatile("vsetivli t0, 2, e8, m1, ta, ma\n\t"
    "vle8.v v8, (%0)\n\t vmv.s.x v0, %3\n\t"
    "vfmerge.vfm v10, v8, %1, v0\n\t vse8.v v10, (%2)"
    :: "r"(V), "f"(S), "r"(O), "r"(m) : "memory", "t0");
  return (uint16_t)(O[0] | (O[1] << 8));
}

static uint16_t i_vfslide1up_vf(uint8_t a, uint8_t b)
{
  uint8_t V[2] = { a, (uint8_t)(a ^ 0x5A) };
  uint8_t O[2] = { FP8VEC_POISON8, FP8VEC_POISON8 };
  double S = bx8(b);
  asm volatile("vsetivli t0, 2, e8, m1, ta, ma\n\t"
    "vle8.v v8, (%0)\n\t vfslide1up.vf v10, v8, %1\n\t vse8.v v10, (%2)"
    :: "r"(V), "f"(S), "r"(O) : "memory", "t0");
  return (uint16_t)(O[0] | (O[1] << 8));
}

static uint16_t i_vfslide1down_vf(uint8_t a, uint8_t b)
{
  uint8_t V[2] = { (uint8_t)(a ^ 0x5A), a };
  uint8_t O[2] = { FP8VEC_POISON8, FP8VEC_POISON8 };
  double S = bx8(b);
  asm volatile("vsetivli t0, 2, e8, m1, ta, ma\n\t"
    "vle8.v v8, (%0)\n\t vfslide1down.vf v10, v8, %1\n\t vse8.v v10, (%2)"
    :: "r"(V), "f"(S), "r"(O) : "memory", "t0");
  return (uint16_t)(O[0] | (O[1] << 8));
}

/* A scalar that is not NaN-boxed for fp8 must read as the canonical NaN;
   this exercises isBoxedF8/unboxF8 rather than the arithmetic. */
static uint8_t i_vfadd_vf_unboxed(uint8_t a, uint64_t sbits, uint8_t *fl)
{
  uint8_t A = a, O = FP8VEC_POISON8;
  double S;
  unsigned long f;
  memcpy(&S, &sbits, 8);
  asm volatile(ASM_HEAD
    "vle8.v v8, (%1)\n\t vfadd.vf v10, v8, %2\n\t vse8.v v10, (%3)\n\t"
    ASM_TAIL
    : "=r"(f) : "r"(&A), "f"(S), "r"(&O) : "memory", "t0");
  *fl = (uint8_t)f;
  return O;
}

/* vl>0 with every element masked off: the reduction takes the vs1 path and,
   for the propagating forms, canonicalizes a NaN there. */
#define DEF_RED_MASKED(fn, MNEM) \
static uint8_t fn(uint8_t s, uint8_t *fl) { \
  uint8_t S = s, O = FP8VEC_POISON8; unsigned long f, zero = 0; \
  asm volatile("csrw fflags, zero\n\t vsetivli t0, 8, e8, m1, ta, ma\n\t" \
    "vmv.v.i v8, 0\n\t vmv.s.x v0, %3\n\t" \
    "vsetivli t0, 1, e8, m1, ta, ma\n\t vle8.v v12, (%1)\n\t" \
    "vsetivli t0, 8, e8, m1, ta, ma\n\t" MNEM " v10, v8, v12, v0.t\n\t" \
    "vsetivli t0, 1, e8, m1, ta, ma\n\t vse8.v v10, (%2)\n\t" ASM_TAIL \
    : "=r"(f) : "r"(&S), "r"(&O), "r"(zero) : "memory", "t0"); \
  *fl = (uint8_t)f; return O; }

DEF_RED_MASKED(i_redusum_masked, "vfredusum.vs")
DEF_RED_MASKED(i_redosum_masked, "vfredosum.vs")

/* --------------------------------------------------------- vector-width run */

/* Same op over a full m8 register group; proves the arm works on every lane
   and under LMUL>1, and that fflags is the OR of the lanes. */
static uint8_t vw_in_a[512], vw_in_b[512], vw_out[512];

static unsigned long vw_add_vv(unsigned long n)
{
  unsigned long f, vl;
  asm volatile("csrw fflags, zero\n\t vsetvli %1, %2, e8, m8, ta, ma\n\t"
    "vle8.v v8, (%3)\n\t vle8.v v16, (%4)\n\t vfadd.vv v24, v8, v16\n\t"
    "vse8.v v24, (%5)\n\t csrr %0, fflags"
    : "=r"(f), "=r"(vl) : "r"(n), "r"(vw_in_a), "r"(vw_in_b), "r"(vw_out)
    : "memory");
  return f | (vl << 8);
}

static unsigned long vw_mul_vv(unsigned long n)
{
  unsigned long f, vl;
  asm volatile("csrw fflags, zero\n\t vsetvli %1, %2, e8, m8, ta, ma\n\t"
    "vle8.v v8, (%3)\n\t vle8.v v16, (%4)\n\t vfmul.vv v24, v8, v16\n\t"
    "vse8.v v24, (%5)\n\t csrr %0, fflags"
    : "=r"(f), "=r"(vl) : "r"(n), "r"(vw_in_a), "r"(vw_in_b), "r"(vw_out)
    : "memory");
  return f | (vl << 8);
}

static unsigned long vw_macc_vv(unsigned long n)
{
  unsigned long f, vl;
  asm volatile("csrw fflags, zero\n\t vsetvli %1, %2, e8, m8, ta, ma\n\t"
    "vle8.v v8, (%3)\n\t vle8.v v16, (%4)\n\t vle8.v v24, (%5)\n\t"
    "vfmacc.vv v24, v8, v16\n\t vse8.v v24, (%5)\n\t csrr %0, fflags"
    : "=r"(f), "=r"(vl) : "r"(n), "r"(vw_in_a), "r"(vw_in_b), "r"(vw_out)
    : "memory");
  return f | (vl << 8);
}

/* ------------------------------------------------------------------ driver */

typedef uint8_t (*fn_bin)(uint8_t, uint8_t, uint8_t *);
typedef uint8_t (*fn_ter)(uint8_t, uint8_t, uint8_t, uint8_t *);
typedef uint8_t (*fn_un)(uint8_t, uint8_t *);
typedef uint16_t (*fn_wbin)(uint8_t, uint8_t, uint8_t *);
typedef uint16_t (*fn_wter)(uint8_t, uint8_t, uint16_t, uint8_t *);
typedef uint16_t (*fn_wun)(uint8_t, uint8_t *);
typedef uint8_t (*fn_nun)(uint16_t, uint8_t *);
typedef uint16_t (*fn_wwbin)(uint16_t, uint8_t, uint8_t *);

static void run_bin(const char *name, fn_bin f, unsigned frm)
{
  set_frm(frm);
  sec_begin(name, 2, 1, 1, 0, 1, (uint8_t)frm, 65536);
  for (unsigned a = 0; a < 256; a++)
    for (unsigned b = 0; b < 256; b++) {
      uint8_t fl, o = f((uint8_t)a, (uint8_t)b, &fl);
      rec2(a, b, o, fl, 1, 1, 1);
    }
}

static void run_ter(const char *name, fn_ter f, unsigned frm)
{
  unsigned i, j, k;
  set_frm(frm);
  sec_begin(name, 3, 1, 1, 1, 1, (uint8_t)frm, NT8 * NT8 * NT8 + 65536);
  for (i = 0; i < NT8; i++)
    for (j = 0; j < NT8; j++)
      for (k = 0; k < NT8; k++) {
        uint8_t fl, o = f(T8[i], T8[j], T8[k], &fl);
        rec3(T8[i], T8[j], T8[k], o, fl, 1, 1, 1, 1);
      }
  rng_s = 0x9E3779B97F4A7C15ULL;
  for (i = 0; i < 65536; i++) {
    uint64_t r = rng();
    uint8_t a = (uint8_t)r, b = (uint8_t)(r >> 8), c = (uint8_t)(r >> 16), fl;
    uint8_t o = f(a, b, c, &fl);
    rec3(a, b, c, o, fl, 1, 1, 1, 1);
  }
}

static void run_un(const char *name, fn_un f, unsigned frm)
{
  set_frm(frm);
  sec_begin(name, 1, 1, 0, 0, 1, (uint8_t)frm, 256);
  for (unsigned a = 0; a < 256; a++) {
    uint8_t fl, o = f((uint8_t)a, &fl);
    rec1(a, o, fl, 1, 1);
  }
}

static void run_wbin(const char *name, fn_wbin f, unsigned frm)
{
  set_frm(frm);
  sec_begin(name, 2, 1, 1, 0, 2, (uint8_t)frm, 65536);
  for (unsigned a = 0; a < 256; a++)
    for (unsigned b = 0; b < 256; b++) {
      uint8_t fl; uint16_t o = f((uint8_t)a, (uint8_t)b, &fl);
      rec2(a, b, o, fl, 1, 1, 2);
    }
}

static void run_wwbin(const char *name, fn_wwbin f, unsigned frm)
{
  unsigned i, b;
  set_frm(frm);
  sec_begin(name, 2, 2, 1, 0, 2, (uint8_t)frm, NT16 * 256);
  for (i = 0; i < NT16; i++)
    for (b = 0; b < 256; b++) {
      uint8_t fl; uint16_t o = f(T16[i], (uint8_t)b, &fl);
      rec2(T16[i], b, o, fl, 2, 1, 2);
    }
}

static void run_wter(const char *name, fn_wter f, unsigned frm)
{
  unsigned i, j, k;
  set_frm(frm);
  sec_begin(name, 3, 1, 1, 2, 2, (uint8_t)frm, NT8 * NT8 * NT16 + 65536);
  for (i = 0; i < NT8; i++)
    for (j = 0; j < NT8; j++)
      for (k = 0; k < NT16; k++) {
        uint8_t fl; uint16_t o = f(T8[i], T8[j], T16[k], &fl);
        rec3(T8[i], T8[j], T16[k], o, fl, 1, 1, 2, 2);
      }
  rng_s = 0xD1B54A32D192ED03ULL;
  for (i = 0; i < 65536; i++) {
    uint64_t r = rng();
    uint8_t a = (uint8_t)r, b = (uint8_t)(r >> 8), fl;
    uint16_t c = (uint16_t)(r >> 16), o = f(a, b, c, &fl);
    rec3(a, b, c, o, fl, 1, 1, 2, 2);
  }
}

static void run_wun(const char *name, fn_wun f, unsigned frm)
{
  set_frm(frm);
  sec_begin(name, 1, 1, 0, 0, 2, (uint8_t)frm, 256);
  for (unsigned a = 0; a < 256; a++) {
    uint8_t fl; uint16_t o = f((uint8_t)a, &fl);
    rec1(a, o, fl, 1, 2);
  }
}

static void run_nun(const char *name, fn_nun f, unsigned frm)
{
  set_frm(frm);
  sec_begin(name, 1, 2, 0, 0, 1, (uint8_t)frm, 65536);
  for (unsigned a = 0; a < 65536; a++) {
    uint8_t fl, o = f((uint16_t)a, &fl);
    rec1(a, o, fl, 2, 1);
  }
}

#define N_RED 20000

static void run_red(const char *name, uint8_t (*f)(const uint8_t *, uint8_t,
                                                   uint8_t *), unsigned frm)
{
  unsigned i, k;
  uint8_t v[RED_VL];
  set_frm(frm);
  sec_begin(name, 2, 8, 1, 0, 1, (uint8_t)frm, N_RED);
  rng_s = 0xC2B2AE3D27D4EB4FULL;
  for (i = 0; i < N_RED; i++) {
    uint64_t packed = 0;
    uint8_t s, fl, o;
    for (k = 0; k < RED_VL; k++)
      v[k] = (i < NT8 * 8) ? T8[(i * RED_VL + k) % NT8] : (uint8_t)rng();
    s = (i < NT8 * 8) ? T8[i % NT8] : (uint8_t)rng();
    for (k = 0; k < RED_VL; k++) packed |= (uint64_t)v[k] << (8 * k);
    o = f(v, s, &fl);
    rec2(packed, s, o, fl, 8, 1, 1);
  }
}

static void run_wred(const char *name, uint16_t (*f)(const uint8_t *, uint16_t,
                                                     uint8_t *), unsigned frm)
{
  unsigned i, k;
  uint8_t v[RED_VL];
  set_frm(frm);
  sec_begin(name, 2, 8, 2, 0, 2, (uint8_t)frm, N_RED);
  rng_s = 0xA24BAED4963EE407ULL;
  for (i = 0; i < N_RED; i++) {
    uint64_t packed = 0;
    uint16_t s, o;
    uint8_t fl;
    for (k = 0; k < RED_VL; k++)
      v[k] = (i < NT8 * 8) ? T8[(i * RED_VL + k) % NT8] : (uint8_t)rng();
    s = (i < NT16 * 8) ? T16[i % NT16] : (uint16_t)rng();
    for (k = 0; k < RED_VL; k++) packed |= (uint64_t)v[k] << (8 * k);
    o = f(v, s, &fl);
    rec2(packed, s, o, fl, 8, 2, 2);
  }
}

int main(void)
{
  unsigned m, i;

  oput(FP8VEC_MAGIC, FP8VEC_MAGIC_LEN);

  run_bin("vfadd.vv", i_vfadd_vv, 0);
  run_bin("vfsub.vv", i_vfsub_vv, 0);
  run_bin("vfmul.vv", i_vfmul_vv, 0);
  run_bin("vfdiv.vv", i_vfdiv_vv, 0);
  run_bin("vfmin.vv", i_vfmin_vv, 0);
  run_bin("vfmax.vv", i_vfmax_vv, 0);
  run_bin("vfsgnj.vv", i_vfsgnj_vv, 0);
  run_bin("vfsgnjn.vv", i_vfsgnjn_vv, 0);
  run_bin("vfsgnjx.vv", i_vfsgnjx_vv, 0);

  run_bin("vfadd.vf", i_vfadd_vf, 0);
  run_bin("vfsub.vf", i_vfsub_vf, 0);
  run_bin("vfrsub.vf", i_vfrsub_vf, 0);
  run_bin("vfmul.vf", i_vfmul_vf, 0);
  run_bin("vfdiv.vf", i_vfdiv_vf, 0);
  run_bin("vfrdiv.vf", i_vfrdiv_vf, 0);
  run_bin("vfmin.vf", i_vfmin_vf, 0);
  run_bin("vfmax.vf", i_vfmax_vf, 0);
  run_bin("vfsgnj.vf", i_vfsgnj_vf, 0);
  run_bin("vfsgnjn.vf", i_vfsgnjn_vf, 0);
  run_bin("vfsgnjx.vf", i_vfsgnjx_vf, 0);

  run_bin("vmfeq.vv", i_vmfeq_vv, 0);
  run_bin("vmfne.vv", i_vmfne_vv, 0);
  run_bin("vmflt.vv", i_vmflt_vv, 0);
  run_bin("vmfle.vv", i_vmfle_vv, 0);
  run_bin("vmfeq.vf", i_vmfeq_vf, 0);
  run_bin("vmfne.vf", i_vmfne_vf, 0);
  run_bin("vmflt.vf", i_vmflt_vf, 0);
  run_bin("vmfle.vf", i_vmfle_vf, 0);
  run_bin("vmfgt.vf", i_vmfgt_vf, 0);
  run_bin("vmfge.vf", i_vmfge_vf, 0);

  run_ter("vfmacc.vv", i_vfmacc_vv, 0);
  run_ter("vfnmacc.vv", i_vfnmacc_vv, 0);
  run_ter("vfmsac.vv", i_vfmsac_vv, 0);
  run_ter("vfnmsac.vv", i_vfnmsac_vv, 0);
  run_ter("vfmadd.vv", i_vfmadd_vv, 0);
  run_ter("vfnmadd.vv", i_vfnmadd_vv, 0);
  run_ter("vfmsub.vv", i_vfmsub_vv, 0);
  run_ter("vfnmsub.vv", i_vfnmsub_vv, 0);
  run_ter("vfmacc.vf", i_vfmacc_vf, 0);
  run_ter("vfnmacc.vf", i_vfnmacc_vf, 0);
  run_ter("vfmsac.vf", i_vfmsac_vf, 0);
  run_ter("vfnmsac.vf", i_vfnmsac_vf, 0);
  run_ter("vfmadd.vf", i_vfmadd_vf, 0);
  run_ter("vfnmadd.vf", i_vfnmadd_vf, 0);
  run_ter("vfmsub.vf", i_vfmsub_vf, 0);
  run_ter("vfnmsub.vf", i_vfnmsub_vf, 0);

  run_un("vfsqrt.v", i_vfsqrt_v, 0);
  run_un("vfrec7.v", i_vfrec7_v, 0);
  run_un("vfrsqrt7.v", i_vfrsqrt7_v, 0);

  run_un("vfcvt.x.f.v", i_vfcvt_x_f_v, 0);
  run_un("vfcvt.xu.f.v", i_vfcvt_xu_f_v, 0);
  run_un("vfcvt.rtz.x.f.v", i_vfcvt_rtz_x_f_v, 0);
  run_un("vfcvt.rtz.xu.f.v", i_vfcvt_rtz_xu_f_v, 0);
  run_un("vfcvt.f.x.v", i_vfcvt_f_x_v, 0);
  run_un("vfcvt.f.xu.v", i_vfcvt_f_xu_v, 0);

  run_wbin("vfwadd.vv", i_vfwadd_vv, 0);
  run_wbin("vfwsub.vv", i_vfwsub_vv, 0);
  run_wbin("vfwmul.vv", i_vfwmul_vv, 0);
  run_wbin("vfwadd.vf", i_vfwadd_vf, 0);
  run_wbin("vfwsub.vf", i_vfwsub_vf, 0);
  run_wbin("vfwmul.vf", i_vfwmul_vf, 0);
  run_wwbin("vfwadd.wv", i_vfwadd_wv, 0);
  run_wwbin("vfwsub.wv", i_vfwsub_wv, 0);
  run_wwbin("vfwadd.wf", i_vfwadd_wf, 0);
  run_wwbin("vfwsub.wf", i_vfwsub_wf, 0);

  run_wter("vfwmacc.vv", i_vfwmacc_vv, 0);
  run_wter("vfwnmacc.vv", i_vfwnmacc_vv, 0);
  run_wter("vfwmsac.vv", i_vfwmsac_vv, 0);
  run_wter("vfwnmsac.vv", i_vfwnmsac_vv, 0);
  run_wter("vfwmacc.vf", i_vfwmacc_vf, 0);
  run_wter("vfwnmacc.vf", i_vfwnmacc_vf, 0);
  run_wter("vfwmsac.vf", i_vfwmsac_vf, 0);
  run_wter("vfwnmsac.vf", i_vfwnmsac_vf, 0);

  run_wun("vfwcvt.f.f.v", i_vfwcvt_f_f_v, 0);
  run_wun("vfwcvt.x.f.v", i_vfwcvt_x_f_v, 0);
  run_wun("vfwcvt.xu.f.v", i_vfwcvt_xu_f_v, 0);
  run_wun("vfwcvt.rtz.x.f.v", i_vfwcvt_rtz_x_f_v, 0);
  run_wun("vfwcvt.rtz.xu.f.v", i_vfwcvt_rtz_xu_f_v, 0);

  run_nun("vfncvt.f.f.w", i_vfncvt_f_f_w, 0);
  run_nun("vfncvt.rod.f.f.w", i_vfncvt_rod_f_f_w, 0);
  run_nun("vfncvt.f.x.w", i_vfncvt_f_x_w, 0);
  run_nun("vfncvt.f.xu.w", i_vfncvt_f_xu_w, 0);

  run_red("vfredosum.vs", i_vfredosum_vs, 0);
  run_red("vfredusum.vs", i_vfredusum_vs, 0);
  run_red("vfredmin.vs", i_vfredmin_vs, 0);
  run_red("vfredmax.vs", i_vfredmax_vs, 0);
  run_wred("vfwredosum.vs", i_vfwredosum_vs, 0);
  run_wred("vfwredusum.vs", i_vfwredusum_vs, 0);

  /* every rounding mode on the ops that actually round */
  for (m = 1; m < 5; m++) {
    char nm[FP8VEC_NAME_LEN];
    static const char *base[] = { "vfadd.vv", "vfsub.vf", "vfmul.vv",
                                  "vfdiv.vv", "vfmacc.vv", "vfmadd.vf",
                                  "vfsqrt.v", "vfncvt.f.f.w", "vfncvt.f.x.w",
                                  "vfncvt.rod.f.f.w", "vfwcvt.x.f.v",
                                  "vfcvt.x.f.v" };
    static const char *tag[] = { "", "@rtz", "@rdn", "@rup", "@rmm" };
    unsigned t;
    for (t = 0; t < sizeof base / sizeof base[0]; t++) {
      strcpy(nm, base[t]); strcat(nm, tag[m]);
      switch (t) {
        case 0: run_bin(nm, i_vfadd_vv, m); break;
        case 1: run_bin(nm, i_vfsub_vf, m); break;
        case 2: run_bin(nm, i_vfmul_vv, m); break;
        case 3: run_bin(nm, i_vfdiv_vv, m); break;
        case 4: run_ter(nm, i_vfmacc_vv, m); break;
        case 5: run_ter(nm, i_vfmadd_vf, m); break;
        case 6: run_un(nm, i_vfsqrt_v, m); break;
        case 7: run_nun(nm, i_vfncvt_f_f_w, m); break;
        case 8: run_nun(nm, i_vfncvt_f_x_w, m); break;
        case 9: run_nun(nm, i_vfncvt_rod_f_f_w, m); break;
        case 10: run_wun(nm, i_vfwcvt_x_f_v, m); break;
        case 11: run_un(nm, i_vfcvt_x_f_v, m); break;
      }
    }
  }
  set_frm(0);

  /* moves, merge and slides */
  sec_begin("vfmv.s.f/f.s", 1, 1, 0, 0, 8, 0, 256);
  for (i = 0; i < 256; i++) {
    uint8_t v; uint64_t u = i_vfmv_roundtrip((uint8_t)i, &v);
    rec1(i, u, v, 1, 8);
  }
  sec_begin("vfmv.v.f", 1, 1, 0, 0, 1, 0, 256);
  for (i = 0; i < 256; i++) rec1(i, i_vfmv_v_f((uint8_t)i), 0, 1, 1);
  sec_begin("vfmerge.vfm", 2, 1, 1, 0, 2, 0, 65536);
  for (i = 0; i < 65536; i++)
    rec2(i >> 8, i & 0xFF, i_vfmerge_vfm((uint8_t)(i >> 8), (uint8_t)i), 0,
         1, 1, 2);
  sec_begin("vfslide1up.vf", 2, 1, 1, 0, 2, 0, 65536);
  for (i = 0; i < 65536; i++)
    rec2(i >> 8, i & 0xFF, i_vfslide1up_vf((uint8_t)(i >> 8), (uint8_t)i), 0,
         1, 1, 2);
  sec_begin("vfslide1down.vf", 2, 1, 1, 0, 2, 0, 65536);
  for (i = 0; i < 65536; i++)
    rec2(i >> 8, i & 0xFF, i_vfslide1down_vf((uint8_t)(i >> 8), (uint8_t)i), 0,
         1, 1, 2);

  /* scalar operands that are not fp8-boxed, and empty-mask reductions */
  {
    static const uint64_t UB[6] = {
      0x3FF0000000000000ULL, 0x0000000000000038ULL, 0xFFFFFFFF00000038ULL,
      0xFFFFFFFFFFFF0038ULL, 0xFFFFFFFFFFFFFE38ULL, 0xFFFFFFFFFFFFFF38ULL,
    };
    unsigned j;
    sec_begin("vfadd.vf.unboxed", 2, 1, 8, 0, 1, 0, 256 * 6);
    for (i = 0; i < 256; i++)
      for (j = 0; j < 6; j++) {
        uint8_t fl, o = i_vfadd_vf_unboxed((uint8_t)i, UB[j], &fl);
        rec2(i, UB[j], o, fl, 1, 8, 1);
      }
    sec_begin("vfredusum.vs.masked", 1, 1, 0, 0, 1, 0, 256);
    for (i = 0; i < 256; i++) {
      uint8_t fl, o = i_redusum_masked((uint8_t)i, &fl);
      rec1(i, o, fl, 1, 1);
    }
    sec_begin("vfredosum.vs.masked", 1, 1, 0, 0, 1, 0, 256);
    for (i = 0; i < 256; i++) {
      uint8_t fl, o = i_redosum_masked((uint8_t)i, &fl);
      rec1(i, o, fl, 1, 1);
    }
  }

  /* LMUL=8 replay: same inputs across all lanes, results must match vl=1 */
  {
    unsigned long r; unsigned chunk;
    sec_begin("lmul8.vfadd.vv", 2, 1, 1, 0, 1, 0, 65536);
    for (chunk = 0; chunk < 65536; chunk += 128) {
      unsigned n = 128, k;
      for (k = 0; k < n; k++) {
        vw_in_a[k] = (uint8_t)((chunk + k) >> 8);
        vw_in_b[k] = (uint8_t)(chunk + k);
        vw_out[k] = FP8VEC_POISON8;
      }
      r = vw_add_vv(n);
      for (k = 0; k < n; k++)
        rec2(vw_in_a[k], vw_in_b[k], vw_out[k],
             (k == 0) ? (uint8_t)(r & 0xFF) : 0, 1, 1, 1);
      if ((r >> 8) != n) { oflush(); return 3; }
    }
    sec_begin("lmul8.vfmul.vv", 2, 1, 1, 0, 1, 0, 65536);
    for (chunk = 0; chunk < 65536; chunk += 128) {
      unsigned n = 128, k;
      for (k = 0; k < n; k++) {
        vw_in_a[k] = (uint8_t)((chunk + k) >> 8);
        vw_in_b[k] = (uint8_t)(chunk + k);
        vw_out[k] = FP8VEC_POISON8;
      }
      r = vw_mul_vv(n);
      for (k = 0; k < n; k++)
        rec2(vw_in_a[k], vw_in_b[k], vw_out[k],
             (k == 0) ? (uint8_t)(r & 0xFF) : 0, 1, 1, 1);
    }
    sec_begin("lmul8.vfmacc.vv", 3, 1, 1, 1, 1, 0, 65536);
    rng_s = 0x27D4EB2F165667C5ULL;
    for (chunk = 0; chunk < 65536; chunk += 128) {
      unsigned n = 128, k;
      uint8_t ca[128];
      for (k = 0; k < n; k++) {
        uint64_t x = rng();
        vw_in_a[k] = (uint8_t)x;
        vw_in_b[k] = (uint8_t)(x >> 8);
        ca[k] = (uint8_t)(x >> 16);
        vw_out[k] = ca[k];
      }
      r = vw_macc_vv(n);
      for (k = 0; k < n; k++)
        rec3(vw_in_a[k], vw_in_b[k], ca[k], vw_out[k],
             (k == 0) ? (uint8_t)(r & 0xFF) : 0, 1, 1, 1, 1);
    }
  }

  oput(FP8VEC_END, FP8VEC_END_LEN);
  oflush();
  return 0;
}
