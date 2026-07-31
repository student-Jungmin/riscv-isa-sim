/* One SEW=8 instruction per build, selected by -DT=n.
   run.sh decides whether each one is supposed to trap; this file only prints
   BEFORE, executes, and prints AFTER if it survived. */

#include <stdio.h>
#include <stdint.h>

static uint8_t a[16];
static uint16_t w[16];

int main(void)
{
  int i;
  for (i = 0; i < 16; i++) { a[i] = (uint8_t)(0x38 + i); w[i] = (uint16_t)i; }
  printf("BEFORE\n");
  fflush(stdout);

#if T == 0
  /* vfclass.v has no e8 arm on purpose: the class mask needs 10 bits */
  asm volatile("vsetivli t0,4,e8,m1,ta,ma\n\t vle8.v v8,(%0)\n\t"
    "vfclass.v v24,v8\n\t vse8.v v24,(%1)"
    :: "r"(a), "r"(a) : "memory", "t0");
#elif T == 1
  asm volatile("vsetivli t0,4,e8,m1,ta,ma\n\t vle8.v v8,(%0)\n\t"
    "vfadd.vv v24,v8,v8\n\t vse8.v v24,(%1)"
    :: "r"(a), "r"(a) : "memory", "t0");
#elif T == 2
  asm volatile("vsetivli t0,4,e8,m1,ta,ma\n\t vle8.v v8,(%0)\n\t"
    "vfwadd.vv v16,v8,v8\n\t vsetivli t0,4,e16,m2,ta,ma\n\t vse16.v v16,(%1)"
    :: "r"(a), "r"(w) : "memory", "t0");
#elif T == 3
  asm volatile("vsetivli t0,4,e16,m2,ta,ma\n\t vle16.v v16,(%0)\n\t"
    "vsetivli t0,4,e8,m1,ta,ma\n\t vfncvt.f.f.w v8,v16\n\t vse8.v v8,(%1)"
    :: "r"(w), "r"(a) : "memory", "t0");
#elif T == 4
  asm volatile("vsetivli t0,4,e8,m1,ta,ma\n\t vle8.v v8,(%0)\n\t"
    "vfwcvt.f.f.v v16,v8\n\t vsetivli t0,4,e16,m2,ta,ma\n\t vse16.v v16,(%1)"
    :: "r"(a), "r"(w) : "memory", "t0");
#elif T == 5
  asm volatile("vsetivli t0,4,e8,m1,ta,ma\n\t vle8.v v8,(%0)\n\t"
    "vfredosum.vs v24,v8,v8\n\t vse8.v v24,(%1)"
    :: "r"(a), "r"(a) : "memory", "t0");
#elif T == 6
  /* not fp8 at all: SEW=8 here is an 8-bit integer beside an f16 */
  asm volatile("vsetivli t0,4,e16,m2,ta,ma\n\t vle16.v v16,(%0)\n\t"
    "vsetivli t0,4,e8,m1,ta,ma\n\t vfncvt.x.f.w v8,v16\n\t vse8.v v8,(%1)"
    :: "r"(w), "r"(a) : "memory", "t0");
#elif T == 7
  asm volatile("vsetivli t0,4,e8,m1,ta,ma\n\t vle8.v v8,(%0)\n\t"
    "vfwcvt.f.x.v v16,v8\n\t vsetivli t0,4,e16,m2,ta,ma\n\t vse16.v v16,(%1)"
    :: "r"(a), "r"(w) : "memory", "t0");
#elif T == 8
  /* no f16 anywhere, yet the loop base still demands Zfh at e8 */
  asm volatile("vsetivli t0,4,e8,m1,ta,ma\n\t vle8.v v8,(%0)\n\t"
    "vfwcvt.x.f.v v16,v8\n\t vsetivli t0,4,e16,m2,ta,ma\n\t vse16.v v16,(%1)"
    :: "r"(a), "r"(w) : "memory", "t0");
#elif T == 9
  asm volatile("vsetivli t0,4,e16,m1,ta,ma\n\t vle16.v v12,(%1)\n\t"
    "vsetivli t0,4,e8,m1,ta,ma\n\t vle8.v v8,(%0)\n\t"
    "vfwredosum.vs v24,v8,v12\n\t vsetivli t0,4,e16,m1,ta,ma\n\t"
    "vse16.v v24,(%1)"
    :: "r"(a), "r"(w) : "memory", "t0");
#else
#error "set -DT=<n>"
#endif

  printf("AFTER\n");
  return 0;
}
