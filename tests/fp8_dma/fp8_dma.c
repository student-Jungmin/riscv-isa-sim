/* Runtime proof for the fp8 DMA path and the E4M3/E5M2 format selector.
   Runs on the spike fp8 fork under pk; every DMA and vector op is executed
   inside the --kernel-addr range so all vector lanes engage. */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#define SPAD_BASE     0xD0000000UL
#define SPAD_PER_LANE 65536UL

#define OFF_A 0u
#define OFF_B 0x100u
#define OFF_C 0x200u

#define DT_INT8 0
#define DT_E4M3 1
#define DT_E5M2 2

extern void k_config(const void *desc);
extern void k_mvin(const void *dram, void *spad);
extern void k_mvout(void *dram, void *spad);
extern void k_vfadd8(void *a, void *b, void *c, long n);
extern void k_vfmul8(void *a, void *b, void *c, long n);
extern void k_vadd8(void *a, void *b, void *c, long n);

typedef struct {
	uint32_t dim_size[4];		/* +0   */
	uint32_t dim_low[4];		/* +16  */
	uint32_t dim_high[4];		/* +32  */
	uint64_t mm_stride[4];		/* +48  */
	uint64_t spad_stride[4];	/* +80  */
	uint16_t element_size;		/* +112 */
	uint16_t vlane_stride;		/* +114 */
	uint8_t  vlane_split_axis;	/* +116 */
	uint8_t  dtype;			/* +117 */
	uint16_t flags;			/* +118 */
	uint64_t indirect_addr;		/* +120 */
	uint16_t indirect_stride;	/* +128 */
	uint16_t indirect_esize;	/* +130 */
	uint8_t  reserved[4];		/* +132 */
	uint64_t fill;			/* +136 */
} desc_t;

_Static_assert(offsetof(desc_t, dim_low) == 16, "dim_low");
_Static_assert(offsetof(desc_t, dim_high) == 32, "dim_high");
_Static_assert(offsetof(desc_t, mm_stride) == 48, "mm_stride");
_Static_assert(offsetof(desc_t, spad_stride) == 80, "spad_stride");
_Static_assert(offsetof(desc_t, element_size) == 112, "element_size");
_Static_assert(offsetof(desc_t, vlane_stride) == 114, "vlane_stride");
_Static_assert(offsetof(desc_t, vlane_split_axis) == 116, "split_axis");
_Static_assert(offsetof(desc_t, dtype) == 117, "dtype");
_Static_assert(offsetof(desc_t, flags) == 118, "flags");
_Static_assert(offsetof(desc_t, indirect_addr) == 120, "indirect_addr");
_Static_assert(offsetof(desc_t, fill) == 136, "fill");

static desc_t desc __attribute__((aligned(64)));

static uint8_t src[256]  __attribute__((aligned(64)));
static uint8_t dst[256]  __attribute__((aligned(64)));
static uint8_t bufA[256] __attribute__((aligned(64)));
static uint8_t bufB[256] __attribute__((aligned(64)));
static uint8_t bufC[256] __attribute__((aligned(64)));

static int lanes;
static unsigned long fails;
static unsigned long checks;

static volatile uint8_t *bank(int lane, unsigned off)
{
	return (volatile uint8_t *)(SPAD_BASE + (unsigned long)lane * SPAD_PER_LANE + off);
}

static void spad_fill(unsigned off, unsigned len, uint8_t v)
{
	for (int l = 0; l < lanes; l++)
		for (unsigned i = 0; i < len; i++)
			*bank(l, off + i) = v;
}

/* Every check funnels through here so a negative control can ask for the
   opposite verdict without duplicating the reporting. */
static int report(const char *tag, int ok, int want_ok, const char *detail)
{
	checks++;
	if (ok == want_ok)
		return 1;
	fails++;
	printf("  FAIL [%s] %s\n", tag, detail);
	return 0;
}

static int cmp_bytes(const char *tag, const uint8_t *got, const uint8_t *want,
		     int n, int want_ok)
{
	int bad = 0, first = -1;
	for (int i = 0; i < n; i++)
		if (got[i] != want[i]) {
			if (first < 0)
				first = i;
			bad++;
		}
	checks++;
	if ((bad == 0) == (want_ok != 0)) {
		if (want_ok)
			printf("  ok   [%s] %d/%d bytes match\n", tag, n, n);
		else
			printf("  ok   [%s] differs as required (%d/%d bytes, first at %d: "
			       "got 0x%02x want 0x%02x)\n", tag, bad, n, first,
			       got[first], want[first]);
		return 1;
	}
	fails++;
	if (want_ok)
		printf("  FAIL [%s] %d/%d bytes differ; first at %d: got 0x%02x expected 0x%02x\n",
		       tag, bad, n, first, got[first], want[first]);
	else
		printf("  FAIL [%s] expected a difference but all %d bytes matched\n", tag, n);
	return bad == 0;
}

/* memset(0xFF) first, then write every field EXCEPT dtype: that is exactly what
   a pre-fp8 producer emits, so dtype keeps the old padding garbage. */
static void desc_build(int split_axis, int d0, int d1, int d2, int d3,
		       const uint64_t *mm, const uint64_t *sp,
		       int esize, int vstride, int set_dtype, uint8_t dtype)
{
	memset(&desc, 0xFF, sizeof desc);
	desc.dim_size[0] = d0; desc.dim_size[1] = d1;
	desc.dim_size[2] = d2; desc.dim_size[3] = d3;
	for (int i = 0; i < 4; i++) {
		desc.dim_low[i] = 0;
		desc.dim_high[i] = desc.dim_size[i];
		desc.mm_stride[i] = mm[i];
		desc.spad_stride[i] = sp[i];
	}
	desc.element_size = esize;
	desc.vlane_stride = vstride;
	desc.vlane_split_axis = split_axis;
	desc.flags = 0;
	desc.indirect_addr = 0;
	desc.indirect_stride = 0;
	desc.indirect_esize = 0;
	desc.fill = 0;
	if (set_dtype)
		desc.dtype = dtype;
}

/*--------------------------------------------------------------------------*/
/* 1. flat fp8 round trip over all 256 byte values                           */
/*--------------------------------------------------------------------------*/
static void test_flat_roundtrip(void)
{
	uint64_t mm[4] = {256, 256, 256, 1};
	uint64_t sp[4] = {256, 256, 256, 1};
	uint8_t seen[256];

	printf("\n=== 1. fp8 flat DMA round trip, all 256 byte values ===\n");
	for (int i = 0; i < 256; i++)
		src[i] = (uint8_t)i;
	memset(dst, 0x5A, sizeof dst);	/* poison: a no-op DMA stays visible */
	spad_fill(OFF_A, 256, 0xA5);

	desc_build(3, 1, 1, 1, 256, mm, sp, 1, 256, 1, DT_E4M3);
	desc.fill = 0x3C;	/* distinctive, so the pad is not confused with poison */
	k_config(&desc);
	k_mvin(src, (void *)(SPAD_BASE + OFF_A));

	for (int i = 0; i < 256; i++)
		seen[i] = *bank(0, OFF_A + i);
	cmp_bytes("mvin->spad(lane0)", seen, src, 256, 1);

	/* vlane_stride spans the whole W axis so used_vlane is 1, but MVIN still
	   walks all n_vu banks and pads the unused ones with desc.fill. */
	for (int l = 1; l < lanes; l++) {
		char tag[48], want[256];
		int i;

		for (i = 0; i < 256; i++)
			seen[i] = *bank(l, OFF_A + i);
		memset(want, 0x3C, 256);
		snprintf(tag, sizeof tag, "lane%d padded with desc.fill", l);
		cmp_bytes(tag, seen, (const uint8_t *)want, 256, 1);
	}

	k_mvout(dst, (void *)(SPAD_BASE + OFF_A));
	cmp_bytes("spad->mvout(dram)", dst, src, 256, 1);

	/* NaN and subnormal encodings are inside the 0..255 sweep; call them out. */
	printf("  note E4M3 NaN 0x7F -> 0x%02x, 0xFF -> 0x%02x; subnormals 0x01 -> 0x%02x, "
	       "0x81 -> 0x%02x\n", dst[0x7F], dst[0xFF], dst[0x01], dst[0x81]);
}

/*--------------------------------------------------------------------------*/
/* 2. lane-banked movement                                                   */
/*--------------------------------------------------------------------------*/
static void test_lane_banked(void)
{
	int w = 256 / lanes;
	uint64_t mm[4] = {256, 256, 0, 1};
	uint64_t sp[4] = {256, 256, 0, 1};
	uint8_t seen[256], flat[256];

	mm[2] = w;
	sp[2] = w;

	printf("\n=== 2. lane-banked DMA (%d lanes x %d bytes, split axis H) ===\n", lanes, w);
	for (int i = 0; i < 256; i++)
		src[i] = (uint8_t)(0xC0 ^ i);
	memset(dst, 0x5A, sizeof dst);
	spad_fill(OFF_A, 256, 0xA5);

	desc_build(2, 1, 1, lanes, w, mm, sp, 1, 1, 1, DT_E4M3);
	k_config(&desc);
	k_mvin(src, (void *)(SPAD_BASE + OFF_A));

	/* Expected banked layout: bank v holds src[v*w .. v*w+w). */
	for (int l = 0; l < lanes; l++)
		for (int i = 0; i < w; i++)
			seen[l * w + i] = *bank(l, OFF_A + i);
	cmp_bytes("banked spad layout", seen, src, 256, 1);

	/* If only one lane had engaged, banks 1.. would still hold the poison. */
	for (int l = 1; l < lanes; l++) {
		char tag[48];
		int stale = 1;
		for (int i = 0; i < w; i++)
			if (*bank(l, OFF_A + i) != 0xA5)
				stale = 0;
		snprintf(tag, sizeof tag, "lane%d engaged", l);
		report(tag, !stale, 1, "bank still holds the 0xA5 poison -> lane never written");
	}
	printf("  ok   all %d banks written (lane %d[0]=0x%02x, expected 0x%02x)\n",
	       lanes, lanes - 1, *bank(lanes - 1, OFF_A), src[(lanes - 1) * w]);

	/* A single-lane DMA would have put src[0..w) in bank0 and nothing else;
	   show the banked result is not that. */
	if (lanes > 1) {
		memcpy(flat, src, 256);
		memset(flat + w, 0xA5, 256 - w);
		cmp_bytes("banked != single-lane", seen, flat, 256, 0);
	}

	k_mvout(dst, (void *)(SPAD_BASE + OFF_A));
	cmp_bytes("banked round trip", dst, src, 256, 1);
}

/* 4-D tile {2,2,lanes,w} split on H: exercises the full N/C/H/W walk with
   non-unit mm strides, not just a contiguous run. */
static void test_lane_banked_4d(void)
{
	int w = 256 / (4 * lanes);
	uint64_t mm[4], sp[4];
	uint8_t seen[256];

	if (w < 1) {
		printf("\n=== 2b skipped: %d lanes leaves no 4-D tile in 256 bytes ===\n", lanes);
		return;
	}
	mm[0] = sp[0] = 2 * lanes * w;
	mm[1] = sp[1] = lanes * w;
	mm[2] = sp[2] = w;
	mm[3] = sp[3] = 1;

	printf("\n=== 2b 4-D banked DMA {2,2,%d,%d}, split axis H ===\n", lanes, w);
	for (int i = 0; i < 256; i++)
		src[i] = (uint8_t)(i * 5 + 1);
	memset(dst, 0x5A, sizeof dst);
	spad_fill(OFF_A, 256, 0xA5);

	desc_build(2, 2, 2, lanes, w, mm, sp, 1, 1, 1, DT_E4M3);
	k_config(&desc);
	k_mvin(src, (void *)(SPAD_BASE + OFF_A));

	{
		int bad = 0, first = -1;
		for (int l = 0; l < lanes; l++)
			for (int n = 0; n < 2; n++)
				for (int c = 0; c < 2; c++)
					for (int i = 0; i < w; i++) {
						int s_off = 2 * w * n + w * c + i;
						int d_idx = (int)(mm[0] * n + mm[1] * c + w * l + i);
						uint8_t got = *bank(l, OFF_A + s_off);
						if (got != src[d_idx]) {
							if (first < 0)
								first = d_idx;
							bad++;
						}
					}
		checks++;
		if (bad) {
			fails++;
			printf("  FAIL [4-D banked layout] %d/256 wrong, first dram idx %d\n",
			       bad, first);
		} else {
			printf("  ok   [4-D banked layout] 256/256 elements at the "
			       "expected bank/offset\n");
		}
	}

	k_mvout(dst, (void *)(SPAD_BASE + OFF_A));
	cmp_bytes("4-D banked round trip", dst, src, 256, 1);
	for (int i = 0; i < 256; i++)
		seen[i] = dst[i];
	(void)seen;
}

/*--------------------------------------------------------------------------*/
/* 3. the E4M3 / E5M2 format selector                                        */
/*--------------------------------------------------------------------------*/
#define NVEC 8

/* Hand-derived oracle. 7 of 8 pairs disagree between the formats; index 5 is a
   deliberate control where both formats give the same bits. */
static const uint8_t va[NVEC]   = {0x38, 0x40, 0x38, 0x30, 0xB8, 0x01, 0x48, 0x44};
static const uint8_t vb[NVEC]   = {0x38, 0x40, 0x40, 0x38, 0x40, 0x01, 0x30, 0x44};
static const uint8_t add_e4[NVEC] = {0x40, 0x48, 0x44, 0x3C, 0x38, 0x02, 0x49, 0x4C};
static const uint8_t add_e5[NVEC] = {0x3C, 0x44, 0x41, 0x39, 0x3E, 0x02, 0x48, 0x48};
static const uint8_t mul_e4[NVEC] = {0x38, 0x48, 0x40, 0x30, 0xC0, 0x00, 0x40, 0x51};
static const uint8_t mul_e5[NVEC] = {0x34, 0x44, 0x3C, 0x2C, 0xBC, 0x00, 0x3C, 0x4C};

static void fmt_setup(uint8_t dtype)
{
	int n = NVEC * lanes;
	uint64_t mm[4] = {0, 0, NVEC, 1};
	uint64_t sp[4] = {0, 0, NVEC, 1};

	mm[0] = mm[1] = n;
	sp[0] = sp[1] = n;
	for (int l = 0; l < lanes; l++)
		for (int i = 0; i < NVEC; i++) {
			bufA[l * NVEC + i] = va[i];
			bufB[l * NVEC + i] = vb[i];
		}
	memset(bufC, 0x5A, sizeof bufC);
	spad_fill(OFF_C, NVEC, 0xAA);

	desc_build(2, 1, 1, lanes, NVEC, mm, sp, 1, 1, 1, dtype);
	k_config(&desc);
	k_mvin(bufA, (void *)(SPAD_BASE + OFF_A));
	k_mvin(bufB, (void *)(SPAD_BASE + OFF_B));
}

/* Runs one op under the format already latched by fmt_setup and checks every
   lane's NVEC results against `want`. */
static int fmt_run(const char *tag, int mul, const uint8_t *want, int want_ok)
{
	uint8_t got[256], exp[256];
	int n = NVEC * lanes;

	if (mul)
		k_vfmul8((void *)(SPAD_BASE + OFF_A), (void *)(SPAD_BASE + OFF_B),
			 (void *)(SPAD_BASE + OFF_C), NVEC);
	else
		k_vfadd8((void *)(SPAD_BASE + OFF_A), (void *)(SPAD_BASE + OFF_B),
			 (void *)(SPAD_BASE + OFF_C), NVEC);
	k_mvout(bufC, (void *)(SPAD_BASE + OFF_C));

	memcpy(got, bufC, n);
	for (int l = 0; l < lanes; l++)
		memcpy(exp + l * NVEC, want, NVEC);
	return cmp_bytes(tag, got, exp, n, want_ok);
}

static void dump_vec(const char *what, const uint8_t *want)
{
	printf("  %s:", what);
	for (int i = 0; i < NVEC; i++)
		printf(" %02x+%02x=%02x", va[i], vb[i], want[i]);
	printf("\n");
}

static void test_format_selector(void)
{
	printf("\n=== 3. E4M3 / E5M2 format selector ===\n");
	dump_vec("expect E4M3", add_e4);
	dump_vec("expect E5M2", add_e5);

	printf(" -- 3a dtype=1 (E4M3)\n");
	fmt_setup(DT_E4M3);
	fmt_run("vfadd E4M3", 0, add_e4, 1);
	fmt_run("vfadd E4M3 != E5M2 table", 0, add_e5, 0);
	fmt_run("vfmul E4M3", 1, mul_e4, 1);

	printf(" -- 3b dtype=2 (E5M2)\n");
	fmt_setup(DT_E5M2);
	fmt_run("vfadd E5M2", 0, add_e5, 1);
	fmt_run("vfadd E5M2 != E4M3 table", 0, add_e4, 0);
	fmt_run("vfmul E5M2", 1, mul_e5, 1);

	/* Leak regression: E5M2 first leaves softfloat_fp8Format at e5m2; if the
	   ALU did not re-inject from byte 117 the next E4M3 op would inherit it. */
	printf(" -- 3c leak regression E5M2 -> E4M3\n");
	fmt_setup(DT_E5M2);
	fmt_run("prime with E5M2", 0, add_e5, 1);
	fmt_setup(DT_E4M3);
	fmt_run("E4M3 after E5M2", 0, add_e4, 1);
	fmt_run("E4M3 after E5M2 not contaminated", 0, add_e5, 0);

	printf(" -- 3d leak regression E4M3 -> E5M2\n");
	fmt_setup(DT_E4M3);
	fmt_run("prime with E4M3", 0, add_e4, 1);
	fmt_setup(DT_E5M2);
	fmt_run("E5M2 after E4M3", 0, add_e5, 1);
	fmt_run("E5M2 after E4M3 not contaminated", 0, add_e4, 0);

	/* Cross-op poison: vfmul under E5M2, then vfadd under E4M3. */
	printf(" -- 3e cross-op poison vfmul(E5M2) -> vfadd(E4M3)\n");
	fmt_setup(DT_E5M2);
	fmt_run("vfmul E5M2 poison", 1, mul_e5, 1);
	fmt_setup(DT_E4M3);
	fmt_run("vfadd E4M3 after poison", 0, add_e4, 1);
}

/*--------------------------------------------------------------------------*/
/* 4. int8 is unaffected                                                     */
/*--------------------------------------------------------------------------*/
static void test_int8(void)
{
	int w = 256 / lanes;
	uint64_t mm[4] = {256, 256, 0, 1};
	uint64_t sp[4] = {256, 256, 0, 1};
	uint8_t seen[256], want[256];

	mm[2] = w;
	sp[2] = w;

	printf("\n=== 4. int8 through the same DMA (dtype=0) ===\n");
	for (int i = 0; i < 256; i++)
		src[i] = (uint8_t)(i * 7 + 13);
	memset(dst, 0x5A, sizeof dst);
	spad_fill(OFF_A, 256, 0xA5);

	desc_build(2, 1, 1, lanes, w, mm, sp, 1, 1, 1, DT_INT8);
	k_config(&desc);
	k_mvin(src, (void *)(SPAD_BASE + OFF_A));
	k_mvout(dst, (void *)(SPAD_BASE + OFF_A));
	cmp_bytes("int8 banked round trip", dst, src, 256, 1);

	/* Integer vector add on the same scratchpad: no softfloat involved. */
	for (int l = 0; l < lanes; l++)
		for (int i = 0; i < NVEC; i++) {
			bufA[l * NVEC + i] = (uint8_t)(0x70 + i);
			bufB[l * NVEC + i] = (uint8_t)(0x30 + i * 3);
		}
	{
		int n = NVEC * lanes;
		uint64_t m2[4] = {0, 0, NVEC, 1};
		uint64_t s2[4] = {0, 0, NVEC, 1};

		m2[0] = m2[1] = n;
		s2[0] = s2[1] = n;
		memset(bufC, 0x5A, sizeof bufC);
		spad_fill(OFF_C, NVEC, 0xAA);
		desc_build(2, 1, 1, lanes, NVEC, m2, s2, 1, 1, 1, DT_INT8);
		k_config(&desc);
		k_mvin(bufA, (void *)(SPAD_BASE + OFF_A));
		k_mvin(bufB, (void *)(SPAD_BASE + OFF_B));
		k_vadd8((void *)(SPAD_BASE + OFF_A), (void *)(SPAD_BASE + OFF_B),
			(void *)(SPAD_BASE + OFF_C), NVEC);
		k_mvout(bufC, (void *)(SPAD_BASE + OFF_C));
		for (int l = 0; l < lanes; l++)
			for (int i = 0; i < NVEC; i++)
				want[l * NVEC + i] =
					(uint8_t)(bufA[l * NVEC + i] + bufB[l * NVEC + i]);
		cmp_bytes("int8 vadd.vv (wrapping)", bufC, want, n, 1);
	}

	/* Byte 117 left as pre-fp8 padding garbage (0xFF): does the int8 DMA care? */
	printf(" -- 4b descriptor byte 117 left as 0xFF garbage\n");
	memset(dst, 0x5A, sizeof dst);
	spad_fill(OFF_A, 256, 0xA5);
	desc_build(2, 1, 1, lanes, w, mm, sp, 1, 1, 0, 0);
	printf("  desc.dtype (byte 117) = 0x%02x\n", desc.dtype);
	k_config(&desc);
	k_mvin(src, (void *)(SPAD_BASE + OFF_A));
	k_mvout(dst, (void *)(SPAD_BASE + OFF_A));
	for (int i = 0; i < 256; i++)
		seen[i] = dst[i];
	cmp_bytes("int8 round trip, byte117=0xFF", seen, src, 256, 1);

	/* Integer vector arithmetic never consults elem_dtype, so garbage in
	   byte 117 must not disturb it either. */
	{
		int n = NVEC * lanes;
		uint64_t m2[4] = {0, 0, NVEC, 1};
		uint64_t s2[4] = {0, 0, NVEC, 1};

		m2[0] = m2[1] = n;
		s2[0] = s2[1] = n;
		for (int l = 0; l < lanes; l++)
			for (int i = 0; i < NVEC; i++) {
				bufA[l * NVEC + i] = (uint8_t)(0x70 + i);
				bufB[l * NVEC + i] = (uint8_t)(0x30 + i * 3);
			}
		memset(bufC, 0x5A, sizeof bufC);
		spad_fill(OFF_C, NVEC, 0xAA);
		desc_build(2, 1, 1, lanes, NVEC, m2, s2, 1, 1, 0, 0);
		k_config(&desc);
		k_mvin(bufA, (void *)(SPAD_BASE + OFF_A));
		k_mvin(bufB, (void *)(SPAD_BASE + OFF_B));
		k_vadd8((void *)(SPAD_BASE + OFF_A), (void *)(SPAD_BASE + OFF_B),
			(void *)(SPAD_BASE + OFF_C), NVEC);
		k_mvout(bufC, (void *)(SPAD_BASE + OFF_C));
		for (int l = 0; l < lanes; l++)
			for (int i = 0; i < NVEC; i++)
				want[l * NVEC + i] =
					(uint8_t)(bufA[l * NVEC + i] + bufB[l * NVEC + i]);
		cmp_bytes("int8 vadd.vv, byte117=0xFF", bufC, want, n, 1);
	}

	/* Same garbage byte, now driving a vector fp8 op: 0xFF is neither 1 nor 2
	   but silently selects E4M3. Recorded, not asserted as desirable. */
	{
		int n = NVEC * lanes;
		uint64_t m2[4] = {0, 0, NVEC, 1};
		uint64_t s2[4] = {0, 0, NVEC, 1};

		m2[0] = m2[1] = n;
		s2[0] = s2[1] = n;
		for (int l = 0; l < lanes; l++)
			for (int i = 0; i < NVEC; i++) {
				bufA[l * NVEC + i] = va[i];
				bufB[l * NVEC + i] = vb[i];
			}
		memset(bufC, 0x5A, sizeof bufC);
		spad_fill(OFF_C, NVEC, 0xAA);
		/* Prime with E5M2 so "no injection" would be visible. */
		desc_build(2, 1, 1, lanes, NVEC, m2, s2, 1, 1, 1, DT_E5M2);
		k_config(&desc);
		k_mvin(bufA, (void *)(SPAD_BASE + OFF_A));
		k_mvin(bufB, (void *)(SPAD_BASE + OFF_B));
		fmt_run("prime E5M2 before garbage", 0, add_e5, 1);
		desc_build(2, 1, 1, lanes, NVEC, m2, s2, 1, 1, 0, 0);
		k_config(&desc);
		fmt_run("byte117=0xFF selects E4M3", 0, add_e4, 1);
	}
}

/*--------------------------------------------------------------------------*/
/* negative controls: prove the harness can fail                             */
/*--------------------------------------------------------------------------*/
static void test_negative_controls(void)
{
	uint64_t mm[4] = {256, 256, 256, 1};
	uint64_t sp[4] = {256, 256, 256, 1};
	uint8_t seen[256], bogus[256];

	printf("\n=== 5. negative controls (each must be detected) ===\n");

	/* N1: no MVIN at all -> MVOUT must not reproduce src. */
	for (int i = 0; i < 256; i++)
		src[i] = (uint8_t)(i ^ 0x3C);
	memset(dst, 0x5A, sizeof dst);
	spad_fill(OFF_A, 256, 0xA5);
	desc_build(3, 1, 1, 1, 256, mm, sp, 1, 256, 1, DT_E4M3);
	k_config(&desc);
	k_mvout(dst, (void *)(SPAD_BASE + OFF_A));
	cmp_bytes("N1 mvout without mvin != src", dst, src, 256, 0);

	/* N2: real round trip, but one source byte perturbed after the DMA. */
	memset(dst, 0x5A, sizeof dst);
	k_mvin(src, (void *)(SPAD_BASE + OFF_A));
	k_mvout(dst, (void *)(SPAD_BASE + OFF_A));
	memcpy(bogus, src, 256);
	bogus[137] ^= 0x01;
	cmp_bytes("N2 perturbed expectation detected", dst, bogus, 256, 0);
	cmp_bytes("N2 unperturbed still passes", dst, src, 256, 1);

	/* N3: an all-zero scratchpad must not pass as a copy of nonzero data. */
	spad_fill(OFF_A, 256, 0x00);
	memset(dst, 0x5A, sizeof dst);
	k_mvout(dst, (void *)(SPAD_BASE + OFF_A));
	for (int i = 0; i < 256; i++)
		seen[i] = dst[i];
	cmp_bytes("N3 zeroed spad != src", seen, src, 256, 0);
	memset(bogus, 0, 256);
	cmp_bytes("N3 zeroed spad really moved zeros", seen, bogus, 256, 1);
}

int main(int argc, char **argv)
{
	lanes = (argc > 1) ? atoi(argv[1]) : 4;
	if (lanes < 1 || lanes > 32 || 256 % lanes) {
		printf("usage: fp8_dma <lanes>  (divisor of 256, <=32)\n");
		return 2;
	}
	printf("fp8 DMA + format selector test, lanes=%d, spad=0x%lx +0x%lx/lane\n",
	       lanes, SPAD_BASE, SPAD_PER_LANE);
	printf("sizeof(desc)=%u\n", (unsigned)sizeof(desc_t));

	test_flat_roundtrip();
	test_lane_banked();
	test_lane_banked_4d();
	test_format_selector();
	test_int8();
	test_negative_controls();

	printf("\n%lu checks, %lu failures\n", checks, fails);
	printf(fails ? "RESULT: FAIL\n" : "RESULT: PASS\n");
	return fails ? 1 : 0;
}
