// End-to-end fp8 matmul on the torchsim systolic array: descriptor -> MVIN ->
// w_vpush/i_vpush/vpop -> MVOUT. Reference values come from ref.py.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "expected.h"

#define SPAD_VADDR 0xD0000000UL
#define W_OFF 0
#define X_OFF 1024
#define Y_OFF 2048

// Flat MVIN/MVOUT descriptor; offsets must match riscv/insns/torchsim_mvin_common.h.
typedef struct {
  uint32_t dim_size[4];      // +0
  uint32_t dim_low[4];       // +16
  uint32_t dim_high[4];      // +32
  uint64_t mm_stride[4];     // +48
  uint64_t spad_stride[4];   // +80
  uint16_t element_size;     // +112
  uint16_t vlane_stride;     // +114
  uint8_t  vlane_split_axis; // +116
  uint8_t  dtype;            // +117
  uint16_t flags;            // +118
  uint64_t indirect_addr;    // +120
  uint16_t indirect_stride;  // +128
  uint16_t indirect_esize;   // +130
  uint16_t pad[2];
  uint64_t fill;             // +136
} desc_t;

typedef struct {
  desc_t *dw, *dx, *dy;
  const void *W, *X;
  void *Y;
  long nvu, R;
  int out32;
} job_t;

static desc_t g_dw, g_dx, g_dy;
static uint8_t g_out[4 * 32 * 32];

#define CONFIG_DESC(p) \
  asm volatile(".insn r 0x2b, 3, 7, x0, %0, x0" :: "r"(p) : "memory")
#define MVIN(dram, spad) \
  asm volatile(".insn r 0x2b, 3, 2, x0, %0, %1" :: "r"(dram), "r"(spad) : "memory")
#define MVOUT(dram, spad) \
  asm volatile(".insn r 0x2b, 3, 3, x0, %0, %1" :: "r"(dram), "r"(spad) : "memory")

// w_vpush v1 = 0x2600305b|(1<<20), i_vpush v2 = 0x2200305b|(2<<20),
// vpop v3 = 0x0800305b|(3<<7), vpop v4 = 0x0800305b|(4<<7).
#define W_VPUSH_V1 ".word 0x2610305b\n\t"
#define I_VPUSH_V2 ".word 0x2220305b\n\t"
#define VPOP_V3    ".word 0x080031db\n\t"
#define VPOP_V4    ".word 0x0800325b\n\t"

__attribute__((noinline, aligned(64)))
void npu_kernel(job_t *j)
{
  volatile long frame[4];
  unsigned long ws = SPAD_VADDR + W_OFF;
  unsigned long xs = SPAD_VADDR + X_OFF;
  unsigned long ys = SPAD_VADDR + Y_OFF;
  long t;
  frame[0] = 0;

  CONFIG_DESC(j->dw);
  MVIN(j->W, ws);
  CONFIG_DESC(j->dx);
  MVIN(j->X, xs);

  // Always push nvu weights per lane: the SA weight deque is a sliding window of
  // sa_dim entries, so a short push would leave the previous case's tail in it.
  if (j->out32) {
    asm volatile(
      "vsetvli %0, %1, e8, m1, ta, ma\n\t"
      "vle8.v v1, (%2)\n\t"
      W_VPUSH_V1
      "vsetvli %0, %3, e8, m1, ta, ma\n\t"
      "vle8.v v2, (%4)\n\t"
      I_VPUSH_V2
      "vsetvli %0, %3, e32, m4, ta, ma\n\t"
      VPOP_V4
      "vse32.v v4, (%5)\n\t"
      : "=&r"(t)
      : "r"(j->nvu), "r"(ws), "r"(j->R), "r"(xs), "r"(ys)
      : "memory");
  } else {
    asm volatile(
      "vsetvli %0, %1, e8, m1, ta, ma\n\t"
      "vle8.v v1, (%2)\n\t"
      W_VPUSH_V1
      "vsetvli %0, %3, e8, m1, ta, ma\n\t"
      "vle8.v v2, (%4)\n\t"
      I_VPUSH_V2
      VPOP_V3
      "vse8.v v3, (%5)\n\t"
      : "=&r"(t)
      : "r"(j->nvu), "r"(ws), "r"(j->R), "r"(xs), "r"(ys)
      : "memory");
  }

  CONFIG_DESC(j->dy);
  MVOUT(j->Y, ys);
  (void)frame[0];
}

static void fill_desc(desc_t *d, int lane_dim, int elt_dim, uint64_t mm_lane,
                      uint64_t mm_elt, uint64_t spad_lane, int esz, int dtype)
{
  memset(d, 0, sizeof(*d));
  d->dim_size[0] = 1; d->dim_size[1] = 1;
  d->dim_size[2] = lane_dim; d->dim_size[3] = elt_dim;
  d->mm_stride[2] = mm_lane; d->mm_stride[3] = mm_elt;
  d->spad_stride[2] = spad_lane; d->spad_stride[3] = 1;
  d->element_size = esz;
  d->vlane_stride = 1;
  d->vlane_split_axis = 2;      // H is the lane axis
  d->dtype = dtype;
}

static float f8_to_float(uint8_t b, int fmt)
{
  int s = (b >> 7) & 1, e, m, mb, bias;
  float v;
  if (fmt == 2) { e = (b >> 2) & 0x1F; m = b & 3; mb = 2; bias = 15; }
  else          { e = (b >> 3) & 0x0F; m = b & 7; mb = 3; bias = 7; }
  if (fmt == 1 && e == 0xF && m == 7) return 0.0f / 0.0f;
  if (fmt == 2 && e == 0x1F) return m ? 0.0f / 0.0f : (s ? -1e30f : 1e30f);
  if (e == 0) v = (float)m / (float)(1 << mb);
  else        v = 1.0f + (float)m / (float)(1 << mb);
  while (e > bias) { v *= 2.0f; e--; }
  while (e < bias) { v /= 2.0f; e++; }
  if (e == 0) v = v;
  return s ? -v : v;
}

static int run_case(const fp8_case_t *c, int nvu, int quiet)
{
  int M = c->M, K = c->K, R = c->R, n = R * M, i, bad = 0;
  job_t job;

  if (nvu < M || nvu < K) {
    printf("SKIP %-18s needs >= %d lanes, have %d\n", c->name,
           M > K ? M : K, nvu);
    return 0;
  }
  memset(g_out, 0xA5, sizeof(g_out));

  // W[j][k]: lane j, spad offset k.  X[i][k]: lane k, spad offset i.
  // Y[i][j]: lane j, spad offset i -> DRAM row-major [R][M].
  fill_desc(&g_dw, M, K, K, 1, K, 1, c->fmt);
  fill_desc(&g_dx, K, R, 1, K, R, 1, c->fmt);
  fill_desc(&g_dy, M, R, 1, M, R, c->out == 32 ? 4 : 1, c->fmt);

  job.dw = &g_dw; job.dx = &g_dx; job.dy = &g_dy;
  job.W = c->W; job.X = c->X; job.Y = g_out;
  job.nvu = nvu; job.R = R; job.out32 = (c->out == 32);
  npu_kernel(&job);

  for (i = 0; i < n; i++) {
    if (c->out == 8) {
      uint8_t got = g_out[i], want = c->e8[i];
      if (got != want) {
        bad++;
        printf("  [%3d] got 0x%02x (%g) want 0x%02x (%g)\n", i, got,
               f8_to_float(got, c->fmt), want, f8_to_float(want, c->fmt));
      }
    } else {
      uint32_t got, want = c->e32[i];
      float gf, wf;
      memcpy(&got, g_out + 4 * i, 4);
      memcpy(&gf, &got, 4); memcpy(&wf, &want, 4);
      if (got != want) {
        bad++;
        printf("  [%3d] got 0x%08x (%g) want 0x%08x (%g)\n", i, got, gf, want, wf);
      }
    }
  }
  printf("%-4s %-18s M=%d K=%d R=%d fmt=%s out=%s lanes=%d  %d/%d ok\n",
         bad ? "FAIL" : "PASS", c->name, M, K, R,
         c->fmt == 1 ? "e4m3" : "e5m2", c->out == 32 ? "fp32" : "fp8 ",
         nvu, n - bad, n);
  if (!quiet && n <= 8) {
    printf("     out:");
    for (i = 0; i < n; i++) {
      if (c->out == 8) printf(" 0x%02x", g_out[i]);
      else { uint32_t v; memcpy(&v, g_out + 4 * i, 4); printf(" 0x%08x", v); }
    }
    printf("\n");
  }
  return bad;
}

// 16*1 + 1*1.5 = 17.5, whose E4M3 neighbours are 16 (0x58) and 18 (0x59):
// RNE gives 0x59, RTZ gives 0x58, so the byte names vpop's rounding mode.
static const uint8_t rp_W[2] = {0x58, 0x38};
static const uint8_t rp_X[2] = {0x38, 0x3C};

static uint8_t rounding_probe(int nvu, int frm, int touch_fp)
{
  job_t job;
  float a = 1.5f, b = 2.5f, z;
  memset(g_out, 0xA5, sizeof(g_out));
  fill_desc(&g_dw, 1, 2, 2, 1, 2, 1, 1);
  fill_desc(&g_dx, 2, 1, 1, 2, 1, 1, 1);
  fill_desc(&g_dy, 1, 1, 1, 1, 1, 1, 1);
  job.dw = &g_dw; job.dx = &g_dx; job.dy = &g_dy;
  job.W = rp_W; job.X = rp_X; job.Y = g_out;
  job.nvu = nvu; job.R = 1; job.out32 = 0;

  asm volatile("csrw frm, %0" :: "r"((unsigned long)frm));
  if (touch_fp)
    asm volatile("fadd.s %0, %1, %2" : "=f"(z) : "f"(a), "f"(b));
  npu_kernel(&job);
  asm volatile("csrw frm, zero");
  return g_out[0];
}

int main(int argc, char **argv)
{
  int nvu = argc > 1 ? atoi(argv[1]) : 32;
  const char *only = argc > 2 ? argv[2] : 0;
  int i, bad = 0, ran = 0;

  printf("fp8 systolic-array e2e, lanes=%d\n", nvu);
  if (only && !strcmp(only, "rounding")) {
    uint8_t r0 = rounding_probe(nvu, 0, 0);   // frm=RNE, no prior FP op
    uint8_t r1 = rounding_probe(nvu, 1, 0);   // frm=RTZ, no prior FP op
    uint8_t r2 = rounding_probe(nvu, 1, 1);   // frm=RTZ, one fadd.s first
    printf("vpop 17.5 -> e4m3: frm=RNE:0x%02x  frm=RTZ:0x%02x  "
           "frm=RTZ+fadd.s:0x%02x\n", r0, r1, r2);
    printf("%s: vpop %s frm (0x59=RNE/18, 0x58=RTZ/16)\n",
           r1 == r2 ? "OK" : "BUG", r1 == r2 ? "honours" : "ignores");
    return r1 != r2;
  }
  for (i = 0; i < N_FP8_CASES; i++) {
    if (only && strcmp(only, fp8_cases[i].name)) continue;
    if (fp8_cases[i].nvu > nvu) continue;
    bad += run_case(&fp8_cases[i], nvu, 0);
    ran++;
  }
  printf("%s: %d case(s), %d mismatching element(s)\n",
         bad ? "FAILED" : "ALL PASS", ran, bad);
  return bad != 0;
}
