/* Host-side reference for the spike fp8 vector stream.
   Each expectation is written from the RVV spec's definition of the
   instruction and evaluated with libsoftfloat, then diffed bit for bit. */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "softfloat.h"
#include "stream.h"

#define MAX_REPORT_DEFAULT 8

static uint64_t g_mutate;   /* corrupt one expectation in N; negative control */
static uint64_t g_seen, g_bad, g_flagbad;
static int g_max_report = MAX_REPORT_DEFAULT;

typedef struct {
  const char *name;
  uint64_t n, val_bad, flag_bad, unchecked;
  uint8_t or_flags;
  int poison;
} secstat;

/* ---------------------------------------------------------------- helpers */

static float8_t F8(uint64_t v) { float8_t r; r.v = (uint8_t)v; return r; }
static float16_t F16(uint64_t v) { float16_t r; r.v = (uint16_t)v; return r; }
static uint8_t neg8(uint64_t v) { return (uint8_t)(v ^ 0x80); }
static float16_t neg16(float16_t a) { a.v ^= 0x8000; return a; }

#define FORMAT_NAN (softfloat_fp8Format == softfloat_fp8_e4m3 ? 0x7F : 0x7E)

static void clr(void) { softfloat_exceptionFlags = 0; }
static uint8_t flg(void) { return (uint8_t)softfloat_exceptionFlags; }

/* ------------------------------------------------------- expectation table */

/* Every entry computes what the RVV spec says the instruction produces.
   Returns 0 when this reference has no opinion, so the section is counted
   as unchecked instead of silently passing. */
typedef int (*expect_fn)(const uint64_t *in, uint64_t *out, uint8_t *fl);

#define EXP(name) static int name(const uint64_t *in, uint64_t *out, uint8_t *fl)
#define A ((uint8_t)in[0])
#define B ((uint8_t)in[1])
#define C ((uint8_t)in[2])
#define DONE(v) do { *out = (v); *fl = flg(); return 1; } while (0)

EXP(e_add_vv) { clr(); DONE(f8_add(F8(B), F8(A)).v); }
EXP(e_sub_vv) { clr(); DONE(f8_sub(F8(A), F8(B)).v); }
EXP(e_mul_vv) { clr(); DONE(f8_mul(F8(B), F8(A)).v); }
EXP(e_div_vv) { clr(); DONE(f8_div(F8(A), F8(B)).v); }
EXP(e_min_vv) { clr(); DONE(f8_min(F8(A), F8(B)).v); }
EXP(e_max_vv) { clr(); DONE(f8_max(F8(A), F8(B)).v); }
EXP(e_sgnj)   { clr(); DONE((uint8_t)((A & 0x7F) | (B & 0x80))); }
EXP(e_sgnjn)  { clr(); DONE((uint8_t)((A & 0x7F) | ((B ^ 0x80) & 0x80))); }
EXP(e_sgnjx)  { clr(); DONE((uint8_t)((A & 0x7F) | ((A ^ B) & 0x80))); }
EXP(e_rsub)   { clr(); DONE(f8_sub(F8(B), F8(A)).v); }
EXP(e_rdiv)   { clr(); DONE(f8_div(F8(B), F8(A)).v); }

EXP(e_eq) { clr(); DONE(f8_eq(F8(A), F8(B))); }
EXP(e_ne) { clr(); DONE(!f8_eq(F8(A), F8(B))); }
EXP(e_lt) { clr(); DONE(f8_lt(F8(A), F8(B))); }
EXP(e_le) { clr(); DONE(f8_le(F8(A), F8(B))); }
EXP(e_gt) { clr(); DONE(f8_lt(F8(B), F8(A))); }
EXP(e_ge) { clr(); DONE(f8_le(F8(B), F8(A))); }

/* vd = +(vs1*vs2) + vd and its seven sign variants; a=vs2, b=vs1, c=vd */
EXP(e_macc)  { clr(); DONE(f8_mulAdd(F8(B), F8(A), F8(C)).v); }
EXP(e_nmacc) { clr(); DONE(f8_mulAdd(F8(neg8(B)), F8(A), F8(neg8(C))).v); }
EXP(e_msac)  { clr(); DONE(f8_mulAdd(F8(B), F8(A), F8(neg8(C))).v); }
EXP(e_nmsac) { clr(); DONE(f8_mulAdd(F8(neg8(B)), F8(A), F8(C)).v); }
EXP(e_madd)  { clr(); DONE(f8_mulAdd(F8(C), F8(B), F8(A)).v); }
EXP(e_nmadd) { clr(); DONE(f8_mulAdd(F8(neg8(C)), F8(B), F8(neg8(A))).v); }
EXP(e_msub)  { clr(); DONE(f8_mulAdd(F8(C), F8(B), F8(neg8(A))).v); }
EXP(e_nmsub) { clr(); DONE(f8_mulAdd(F8(neg8(C)), F8(B), F8(A)).v); }

EXP(e_sqrt)   { clr(); DONE(f8_sqrt(F8(A)).v); }
EXP(e_rec7)   { clr(); DONE(f8_recip7(F8(A)).v); }
EXP(e_rsqrt7) { clr(); DONE(f8_rsqrte7(F8(A)).v); }

EXP(e_cvt_x_f)      { clr(); DONE((uint8_t)f8_to_i8(F8(A), softfloat_roundingMode, true)); }
EXP(e_cvt_xu_f)     { clr(); DONE((uint8_t)f8_to_ui8(F8(A), softfloat_roundingMode, true)); }
EXP(e_cvt_rtz_x_f)  { clr(); DONE((uint8_t)f8_to_i8(F8(A), softfloat_round_minMag, true)); }
EXP(e_cvt_rtz_xu_f) { clr(); DONE((uint8_t)f8_to_ui8(F8(A), softfloat_round_minMag, true)); }
EXP(e_cvt_f_x)      { clr(); DONE(i32_to_f8((int8_t)A).v); }
EXP(e_cvt_f_xu)     { clr(); DONE(ui32_to_f8(A).v); }

static float16_t w8(uint8_t v) { return f8_to_f16(F8(v)); }

EXP(e_wadd_vv) { clr(); DONE(f16_add(w8(A), w8(B)).v); }
EXP(e_wsub_vv) { clr(); DONE(f16_sub(w8(A), w8(B)).v); }
EXP(e_wmul_vv) { clr(); DONE(f16_mul(w8(A), w8(B)).v); }
EXP(e_wadd_wv) { clr(); DONE(f16_add(F16(in[0]), w8(B)).v); }
EXP(e_wsub_wv) { clr(); DONE(f16_sub(F16(in[0]), w8(B)).v); }

EXP(e_wmacc)  { clr(); DONE(f16_mulAdd(w8(B), w8(A), F16(in[2])).v); }
EXP(e_wnmacc) { clr(); DONE(f16_mulAdd(neg16(w8(B)), w8(A), neg16(F16(in[2]))).v); }
EXP(e_wmsac)  { clr(); DONE(f16_mulAdd(w8(B), w8(A), neg16(F16(in[2]))).v); }
EXP(e_wnmsac) { clr(); DONE(f16_mulAdd(neg16(w8(B)), w8(A), F16(in[2])).v); }

EXP(e_wcvt_f_f)      { clr(); DONE(f8_to_f16(F8(A)).v); }
EXP(e_wcvt_x_f)      { clr(); DONE((uint16_t)f8_to_i16(F8(A), softfloat_roundingMode, true)); }
EXP(e_wcvt_xu_f)     { clr(); DONE((uint16_t)f8_to_ui16(F8(A), softfloat_roundingMode, true)); }
EXP(e_wcvt_rtz_x_f)  { clr(); DONE((uint16_t)f8_to_i16(F8(A), softfloat_round_minMag, true)); }
EXP(e_wcvt_rtz_xu_f) { clr(); DONE((uint16_t)f8_to_ui16(F8(A), softfloat_round_minMag, true)); }

EXP(e_ncvt_f_f) { clr(); DONE(f16_to_f8(F16(in[0])).v); }
EXP(e_ncvt_rod)
{
  uint_fast8_t save = softfloat_roundingMode;
  clr();
  softfloat_roundingMode = softfloat_round_odd;
  *out = f16_to_f8(F16(in[0])).v;
  softfloat_roundingMode = save;
  *fl = flg();
  return 1;
}
EXP(e_ncvt_f_x)  { clr(); DONE(i32_to_f8((int16_t)in[0]).v); }
EXP(e_ncvt_f_xu) { clr(); DONE(ui32_to_f8((uint16_t)in[0]).v); }

/* Reductions: in[0] packs the 8 fp8 elements, in[1] is the scalar start. */
#define RED_BODY(OP) \
  int k; float8_t acc = F8(in[1]); \
  clr(); \
  for (k = 0; k < 8; k++) acc = OP(acc, F8(in[0] >> (8 * k))); \
  DONE(acc.v)

EXP(e_redosum) { RED_BODY(f8_add); }
EXP(e_redusum) { RED_BODY(f8_add); }
EXP(e_redmin)  { RED_BODY(f8_min); }
EXP(e_redmax)  { RED_BODY(f8_max); }

#define WRED_BODY(OP) \
  int k; float16_t acc = F16(in[1]); \
  clr(); \
  for (k = 0; k < 8; k++) acc = OP(acc, w8((uint8_t)(in[0] >> (8 * k)))); \
  DONE(acc.v)

EXP(e_wredosum) { WRED_BODY(f16_add); }
EXP(e_wredusum) { WRED_BODY(f16_add); }

/* vfmv.s.f then vfmv.f.s: the scalar comes back NaN-boxed and unchanged. */
EXP(e_mv_roundtrip)
{
  clr();
  *fl = (uint8_t)A;
  *out = 0xFFFFFFFFFFFFFF00ULL | A;
  return 1;
}
EXP(e_mv_v_f) { clr(); DONE(A); }
/* mask bit 0 takes the scalar, bit 1 keeps vs2 */
EXP(e_merge) { clr(); DONE((uint16_t)(B | (A << 8))); }
EXP(e_slide1up) { clr(); DONE((uint16_t)(B | (A << 8))); }
EXP(e_slide1down) { clr(); DONE((uint16_t)(A | (B << 8))); }

/* Only bits 63:8 all ones is a valid fp8 box; anything else reads as the
   canonical NaN, so the sum is that NaN too. */
EXP(e_add_unboxed)
{
  uint8_t s = (in[1] >> 8) == 0x00FFFFFFFFFFFFFFULL
              ? (uint8_t)in[1] : (uint8_t)FORMAT_NAN;
  clr();
  DONE(f8_add(F8(s), F8(A)).v);
}

/* No element is active, so the result is vs1[0]; the propagating forms
   canonicalize a NaN there, the ordered form passes it through. */
EXP(e_red_masked_prop)
{
  clr();
  DONE(((A & 0x7F) == 0x7F && softfloat_fp8Format == softfloat_fp8_e4m3)
       ? (uint64_t)FORMAT_NAN : A);
}
EXP(e_red_masked_plain) { clr(); DONE(A); }

/* ------------------------------------------------------------- dispatch */

/* FLAGS_PER: the record's flags byte is that case's fflags.
   FLAGS_NONE: the field is unused or carries something else.
   FLAGS_CHUNK: lane 0 of each 128-record group holds the group's OR. */
enum { FLAGS_PER, FLAGS_NONE, FLAGS_CHUNK, FLAGS_ELEM };

static const struct { const char *name; expect_fn fn; int fmode; } TAB[] = {
  { "vfadd.vv", e_add_vv, FLAGS_PER }, { "vfsub.vv", e_sub_vv, FLAGS_PER },
  { "vfmul.vv", e_mul_vv, FLAGS_PER }, { "vfdiv.vv", e_div_vv, FLAGS_PER },
  { "vfmin.vv", e_min_vv, FLAGS_PER }, { "vfmax.vv", e_max_vv, FLAGS_PER },
  { "vfsgnj.vv", e_sgnj, FLAGS_PER }, { "vfsgnjn.vv", e_sgnjn, FLAGS_PER },
  { "vfsgnjx.vv", e_sgnjx, FLAGS_PER },
  { "vfadd.vf", e_add_vv, FLAGS_PER }, { "vfsub.vf", e_sub_vv, FLAGS_PER },
  { "vfrsub.vf", e_rsub, FLAGS_PER }, { "vfmul.vf", e_mul_vv, FLAGS_PER },
  { "vfdiv.vf", e_div_vv, FLAGS_PER }, { "vfrdiv.vf", e_rdiv, FLAGS_PER },
  { "vfmin.vf", e_min_vv, FLAGS_PER }, { "vfmax.vf", e_max_vv, FLAGS_PER },
  { "vfsgnj.vf", e_sgnj, FLAGS_PER }, { "vfsgnjn.vf", e_sgnjn, FLAGS_PER },
  { "vfsgnjx.vf", e_sgnjx, FLAGS_PER },
  { "vmfeq.vv", e_eq, FLAGS_PER }, { "vmfne.vv", e_ne, FLAGS_PER },
  { "vmflt.vv", e_lt, FLAGS_PER }, { "vmfle.vv", e_le, FLAGS_PER },
  { "vmfeq.vf", e_eq, FLAGS_PER }, { "vmfne.vf", e_ne, FLAGS_PER },
  { "vmflt.vf", e_lt, FLAGS_PER }, { "vmfle.vf", e_le, FLAGS_PER },
  { "vmfgt.vf", e_gt, FLAGS_PER }, { "vmfge.vf", e_ge, FLAGS_PER },
  { "vfmacc.vv", e_macc, FLAGS_PER }, { "vfnmacc.vv", e_nmacc, FLAGS_PER },
  { "vfmsac.vv", e_msac, FLAGS_PER }, { "vfnmsac.vv", e_nmsac, FLAGS_PER },
  { "vfmadd.vv", e_madd, FLAGS_PER }, { "vfnmadd.vv", e_nmadd, FLAGS_PER },
  { "vfmsub.vv", e_msub, FLAGS_PER }, { "vfnmsub.vv", e_nmsub, FLAGS_PER },
  { "vfmacc.vf", e_macc, FLAGS_PER }, { "vfnmacc.vf", e_nmacc, FLAGS_PER },
  { "vfmsac.vf", e_msac, FLAGS_PER }, { "vfnmsac.vf", e_nmsac, FLAGS_PER },
  { "vfmadd.vf", e_madd, FLAGS_PER }, { "vfnmadd.vf", e_nmadd, FLAGS_PER },
  { "vfmsub.vf", e_msub, FLAGS_PER }, { "vfnmsub.vf", e_nmsub, FLAGS_PER },
  { "vfsqrt.v", e_sqrt, FLAGS_PER }, { "vfrec7.v", e_rec7, FLAGS_PER },
  { "vfrsqrt7.v", e_rsqrt7, FLAGS_PER },
  { "vfcvt.x.f.v", e_cvt_x_f, FLAGS_PER }, { "vfcvt.xu.f.v", e_cvt_xu_f, FLAGS_PER },
  { "vfcvt.rtz.x.f.v", e_cvt_rtz_x_f, FLAGS_PER },
  { "vfcvt.rtz.xu.f.v", e_cvt_rtz_xu_f, FLAGS_PER },
  { "vfcvt.f.x.v", e_cvt_f_x, FLAGS_PER }, { "vfcvt.f.xu.v", e_cvt_f_xu, FLAGS_PER },
  { "vfwadd.vv", e_wadd_vv, FLAGS_PER }, { "vfwsub.vv", e_wsub_vv, FLAGS_PER },
  { "vfwmul.vv", e_wmul_vv, FLAGS_PER },
  { "vfwadd.vf", e_wadd_vv, FLAGS_PER }, { "vfwsub.vf", e_wsub_vv, FLAGS_PER },
  { "vfwmul.vf", e_wmul_vv, FLAGS_PER },
  { "vfwadd.wv", e_wadd_wv, FLAGS_PER }, { "vfwsub.wv", e_wsub_wv, FLAGS_PER },
  { "vfwadd.wf", e_wadd_wv, FLAGS_PER }, { "vfwsub.wf", e_wsub_wv, FLAGS_PER },
  { "vfwmacc.vv", e_wmacc, FLAGS_PER }, { "vfwnmacc.vv", e_wnmacc, FLAGS_PER },
  { "vfwmsac.vv", e_wmsac, FLAGS_PER }, { "vfwnmsac.vv", e_wnmsac, FLAGS_PER },
  { "vfwmacc.vf", e_wmacc, FLAGS_PER }, { "vfwnmacc.vf", e_wnmacc, FLAGS_PER },
  { "vfwmsac.vf", e_wmsac, FLAGS_PER }, { "vfwnmsac.vf", e_wnmsac, FLAGS_PER },
  { "vfwcvt.f.f.v", e_wcvt_f_f, FLAGS_PER }, { "vfwcvt.x.f.v", e_wcvt_x_f, FLAGS_PER },
  { "vfwcvt.xu.f.v", e_wcvt_xu_f, FLAGS_PER },
  { "vfwcvt.rtz.x.f.v", e_wcvt_rtz_x_f, FLAGS_PER },
  { "vfwcvt.rtz.xu.f.v", e_wcvt_rtz_xu_f, FLAGS_PER },
  { "vfncvt.f.f.w", e_ncvt_f_f, FLAGS_PER }, { "vfncvt.rod.f.f.w", e_ncvt_rod, FLAGS_PER },
  { "vfncvt.f.x.w", e_ncvt_f_x, FLAGS_PER }, { "vfncvt.f.xu.w", e_ncvt_f_xu, FLAGS_PER },
  { "vfredosum.vs", e_redosum, FLAGS_PER }, { "vfredusum.vs", e_redusum, FLAGS_PER },
  { "vfredmin.vs", e_redmin, FLAGS_PER }, { "vfredmax.vs", e_redmax, FLAGS_PER },
  { "vfwredosum.vs", e_wredosum, FLAGS_PER }, { "vfwredusum.vs", e_wredusum, FLAGS_PER },
  { "vfmv.s.f/f.s", e_mv_roundtrip, FLAGS_ELEM },
  { "vfmv.v.f", e_mv_v_f, FLAGS_NONE },
  { "vfmerge.vfm", e_merge, FLAGS_NONE },
  { "vfslide1up.vf", e_slide1up, FLAGS_NONE },
  { "vfslide1down.vf", e_slide1down, FLAGS_NONE },
  { "lmul8.vfadd.vv", e_add_vv, FLAGS_CHUNK },
  { "lmul8.vfmul.vv", e_mul_vv, FLAGS_CHUNK },
  { "lmul8.vfmacc.vv", e_macc, FLAGS_CHUNK },
  { "vfadd.vf.unboxed", e_add_unboxed, FLAGS_PER },
  { "vfredusum.vs.masked", e_red_masked_prop, FLAGS_PER },
  { "vfredosum.vs.masked", e_red_masked_plain, FLAGS_PER },
};

/* '@rtz' and friends are a rounding-mode suffix on an otherwise known name */
static expect_fn lookup(const char *name, int *fmode)
{
  char base[FP8VEC_NAME_LEN];
  const char *at = strchr(name, '@');
  size_t i, n = at ? (size_t)(at - name) : strlen(name);
  if (n >= sizeof base) return NULL;
  memcpy(base, name, n);
  base[n] = 0;
  for (i = 0; i < sizeof TAB / sizeof TAB[0]; i++)
    if (!strcmp(TAB[i].name, base)) { *fmode = TAB[i].fmode; return TAB[i].fn; }
  return NULL;
}

/* --------------------------------------------------------------- checking */

static uint64_t rd(const uint8_t *p, int w)
{
  uint64_t v = 0;
  int i;
  for (i = 0; i < w; i++) v |= (uint64_t)p[i] << (8 * i);
  return v;
}

/* The fp8 sections carry a value whose NaN encoding is canonical on both
   sides, so an exact compare is right; nothing here is NaN-tolerant. */
static int check_section(FILE *f, const fp8vec_hdr *h, secstat *st)
{
  expect_fn fn;
  int fmode = FLAGS_PER;
  uint8_t chunk_want = 0, chunk_got = 0;
  uint32_t k;
  int rw = h->in_w[0] + h->in_w[1] + h->in_w[2] + h->out_w + 1;
  uint8_t *buf = malloc((size_t)rw * h->count);
  int reported = 0;
  uint64_t mutcount = 0;

  if (!buf) { fprintf(stderr, "oom\n"); return -1; }
  if (fread(buf, rw, h->count, f) != h->count) {
    fprintf(stderr, "short read in section %s\n", h->name);
    free(buf);
    return -1;
  }

  st->name = strdup(h->name);
  st->n = h->count;
  fn = lookup(h->name, &fmode);
  if (!fn) { st->unchecked = h->count; free(buf); return 0; }

  softfloat_roundingMode = h->frm;
  for (k = 0; k < h->count; k++) {
    const uint8_t *p = buf + (size_t)rw * k;
    uint64_t in[3] = { 0, 0, 0 }, got, want = 0;
    uint8_t gflags, wflags = 0;
    int off = 0, i;

    for (i = 0; i < 3; i++) {
      if (!h->in_w[i]) break;
      in[i] = rd(p + off, h->in_w[i]);
      off += h->in_w[i];
    }
    got = rd(p + off, h->out_w);
    gflags = p[off + h->out_w];

    softfloat_roundingMode = h->frm;
    if (!fn(in, &want, &wflags)) continue;
    if (h->out_w < 8) want &= (1ULL << (8 * h->out_w)) - 1;

    /* negative control: corrupt one expectation in N, values and flags
       alternately, so a silently-passing comparison shows up */
    if (g_mutate && (++mutcount % g_mutate) == 0) {
      if (mutcount % (2 * g_mutate)) want ^= 1; else wflags ^= 1;
    }

    st->or_flags |= gflags;
    if (h->out_w == 1 && got == FP8VEC_POISON8 && want != FP8VEC_POISON8)
      st->poison++;

    if (got != want) {
      st->val_bad++;
      if (reported < g_max_report) {
        reported++;
        fprintf(stderr,
                "MISMATCH %-20s frm=%u in=[%llx,%llx,%llx] got=0x%llx "
                "want=0x%llx\n", h->name, h->frm,
                (unsigned long long)in[0], (unsigned long long)in[1],
                (unsigned long long)in[2], (unsigned long long)got,
                (unsigned long long)want);
      }
    }
    if (fmode == FLAGS_CHUNK) {
      if ((k & 127) == 0) { chunk_got = gflags; chunk_want = 0; }
      chunk_want |= wflags;
      if ((k & 127) == 127 && chunk_got != chunk_want) {
        st->flag_bad++;
        if (reported < g_max_report) {
          reported++;
          fprintf(stderr, "FFLAGS-OR %-18s chunk ending %u got=0x%02x "
                  "want=0x%02x\n", h->name, k, chunk_got, chunk_want);
        }
      }
      continue;
    }
    if (fmode == FLAGS_NONE) continue;
    if (gflags != wflags) {
      st->flag_bad++;
      if (reported < g_max_report) {
        reported++;
        fprintf(stderr,
                "FFLAGS   %-20s frm=%u in=[%llx,%llx,%llx] got=0x%02x "
                "want=0x%02x\n", h->name, h->frm,
                (unsigned long long)in[0], (unsigned long long)in[1],
                (unsigned long long)in[2], gflags, wflags);
      }
    }
  }
  free(buf);
  return 0;
}

int main(int argc, char **argv)
{
  FILE *f;
  char magic[FP8VEC_MAGIC_LEN];
  secstat *stats = NULL;
  size_t nstat = 0, cap = 0, i;
  const char *env;

  if (argc < 2) { fprintf(stderr, "usage: ref <stream> [e5m2]\n"); return 2; }
  f = fopen(argv[1], "rb");
  if (!f) { perror(argv[1]); return 2; }
  if (argc > 2 && !strcmp(argv[2], "e5m2"))
    softfloat_fp8Format = softfloat_fp8_e5m2;

  env = getenv("FP8V_MUTATE");
  if (env) g_mutate = strtoull(env, NULL, 0);
  env = getenv("FP8V_MAX_REPORT");
  if (env) g_max_report = atoi(env);

  if (fread(magic, 1, FP8VEC_MAGIC_LEN, f) != FP8VEC_MAGIC_LEN
      || memcmp(magic, FP8VEC_MAGIC, FP8VEC_MAGIC_LEN)) {
    fprintf(stderr, "bad magic\n");
    return 2;
  }

  for (;;) {
    fp8vec_hdr h;
    long pos = ftell(f);
    char tail[FP8VEC_END_LEN];
    if (fread(&h, 1, sizeof h, f) != sizeof h) break;
    if (!memcmp(&h, FP8VEC_END, FP8VEC_END_LEN)) {
      fseek(f, pos, SEEK_SET);
      if (fread(tail, 1, FP8VEC_END_LEN, f) == FP8VEC_END_LEN)
        fprintf(stderr, "stream terminated cleanly\n");
      break;
    }
    if (nstat == cap) {
      cap = cap ? cap * 2 : 64;
      stats = realloc(stats, cap * sizeof *stats);
    }
    memset(&stats[nstat], 0, sizeof stats[nstat]);
    if (check_section(f, &h, &stats[nstat])) return 2;
    nstat++;
  }

  printf("%-24s %10s %10s %10s %8s %6s\n", "section", "cases", "val_bad",
         "flag_bad", "or_fflag", "state");
  for (i = 0; i < nstat; i++) {
    secstat *s = &stats[i];
    const char *state = s->unchecked ? "SKIP"
                        : (s->val_bad || s->flag_bad) ? "FAIL" : "ok";
    g_seen += s->n;
    g_bad += s->val_bad;
    g_flagbad += s->flag_bad;
    printf("%-24s %10llu %10llu %10llu     0x%02x %6s%s\n", s->name,
           (unsigned long long)s->n, (unsigned long long)s->val_bad,
           (unsigned long long)s->flag_bad, s->or_flags, state,
           s->poison ? "  POISON!" : "");
  }
  printf("\nTOTAL sections=%zu cases=%llu value_mismatch=%llu "
         "fflags_mismatch=%llu\n", nstat, (unsigned long long)g_seen,
         (unsigned long long)g_bad, (unsigned long long)g_flagbad);
  printf("RESULT: %s\n", (g_bad || g_flagbad) ? "FAIL" : "PASS");
  return (g_bad || g_flagbad) ? 1 : 0;
}
