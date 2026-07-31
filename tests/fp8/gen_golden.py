#!/usr/bin/env python3
"""Emit the fp8 golden file that tests/fp8/test_fp8.c diffs against.
torch is the value reference for round-to-nearest-even; fp8_oracle supplies the
exception flags, and both value and flags for the other rounding modes."""

import argparse
import os
import struct
import sys

import numpy as np
import torch

import fp8_oracle as ORACLE

MAGIC = b"FP8GOLD\x01"
NAME_LEN = 32
REC_LEN = 20

OP_WIDEN_F32 = 1
OP_ADD = 2
OP_SUB = 3
OP_MUL = 4
OP_DIV = 5
OP_EQ = 6
OP_LE = 7
OP_LT = 8
OP_NARROW_F8 = 9

# fmt codes are the softfloat_fp8Format enum verbatim
FORMATS = (("e4m3", 0, "float8_e4m3fn"), ("e5m2", 1, "float8_e5m2"))

REC_DTYPE = np.dtype(
    [
        ("in0", "<u4"),
        ("in1", "<u4"),
        ("in2", "<u4"),
        ("out", "<u4"),
        ("flags", "u1"),
        ("known", "u1"),
        ("rmode", "u1"),
        ("rsvd", "u1"),
    ]
)
assert REC_DTYPE.itemsize == REC_LEN

RNG_SEED = 20260730
N_RANDOM_LINEAR = 4096
N_RANDOM_LOG = 4096


# ---------------------------------------------------------------------------
# exception-flag hook (Phase 4)
# ---------------------------------------------------------------------------

BINOPS = {
    OP_ADD: ORACLE.op_add,
    OP_SUB: ORACLE.op_sub,
    OP_MUL: ORACLE.op_mul,
    OP_DIV: ORACLE.op_div,
}

CMPOPS = {OP_EQ: False, OP_LE: True, OP_LT: True}


def flags_for(op, fmt, in0, in1, mode):
    """softfloat_exceptionFlags goldens, straight from the exact-rational
    oracle.  'known' is 1 for every op the oracle covers, which turns the
    driver's flag comparison on."""
    count = len(in0)
    flags = np.zeros(count, dtype="u1")
    known = np.ones(count, dtype="u1")

    if op in BINOPS:
        fn = BINOPS[op]
        for k in range(count):
            flags[k] = fn(int(in0[k]), int(in1[k]), fmt, mode)[1]
    elif op == OP_NARROW_F8:
        for k in range(count):
            flags[k] = ORACLE.op_narrow(int(in0[k]), fmt, mode)[1]
    elif op == OP_WIDEN_F32:
        for k in range(count):
            flags[k] = ORACLE.op_widen(int(in0[k]), fmt)
    elif op in CMPOPS:
        sig = CMPOPS[op]
        for k in range(count):
            flags[k] = ORACLE.cmp_flags(int(in0[k]), int(in1[k]), fmt, sig)
    else:
        known[:] = 0
    return flags, known


def oracle_values(op, fmt, in0, in1, mode):
    """The oracle's own result bits, used as the golden for the rounding modes
    torch cannot produce and as a cross-check against torch for RNE."""
    count = len(in0)
    out = np.zeros(count, dtype="<u4")
    if op in BINOPS:
        fn = BINOPS[op]
        for k in range(count):
            out[k] = fn(int(in0[k]), int(in1[k]), fmt, mode)[0]
    elif op == OP_NARROW_F8:
        for k in range(count):
            out[k] = ORACLE.op_narrow(int(in0[k]), fmt, mode)[0]
    return out


# ---------------------------------------------------------------------------
# bit helpers
# ---------------------------------------------------------------------------

def f8_bits(t):
    return t.contiguous().view(torch.uint8).numpy().astype("<u4")


def f32_bits(t):
    return t.contiguous().view(torch.int32).numpy().view("<u4")


def np_f32_bits(a):
    return np.ascontiguousarray(a, dtype=np.float32).view("<u4")


def make_records(in0, in1, in2, out, op, fmt, mode=ORACLE.RNE):
    count = len(out)
    rec = np.zeros(count, dtype=REC_DTYPE)
    rec["in0"] = in0
    rec["in1"] = in1 if in1 is not None else 0
    rec["in2"] = in2 if in2 is not None else 0
    rec["out"] = out
    rec["rmode"] = mode
    zero = np.zeros(count, dtype="<u4")
    rec["flags"], rec["known"] = flags_for(
        op, fmt, in0, in1 if in1 is not None else zero, mode)
    return rec


# ---------------------------------------------------------------------------
# narrowing input set
# ---------------------------------------------------------------------------

def narrow_inputs(dt):
    """Dense f32 probe set: every fp8 value, every midpoint between adjacent
    fp8 values plus its two f32 neighbours, the overflow boundary, the
    subnormal range, signed zeros, infinities, NaNs and seeded randoms."""
    wide = torch.arange(256, dtype=torch.uint8).view(dt).to(torch.float64).numpy()
    finite = np.unique(wide[np.isfinite(wide)])
    f32fin = finite.astype(np.float32)

    parts = [f32fin]

    mid = ((finite[:-1] + finite[1:]) * 0.5).astype(np.float32)
    parts.append(mid)
    parts.append(np.nextafter(mid, np.float32(-np.inf)))
    parts.append(np.nextafter(mid, np.float32(np.inf)))

    parts.append(np.nextafter(f32fin, np.float32(-np.inf)))
    parts.append(np.nextafter(f32fin, np.float32(np.inf)))

    # the slot just past max finite exists in no fp8 encoding, so add it by hand
    maxfin = finite[-1]
    step = finite[-1] - finite[-2]
    beyond = np.array([maxfin + step, maxfin + 0.5 * step, maxfin + 2.0 * step],
                      dtype=np.float64)
    beyond = np.concatenate([beyond, -beyond]).astype(np.float32)
    parts.append(beyond)
    parts.append(np.nextafter(beyond, np.float32(-np.inf)))
    parts.append(np.nextafter(beyond, np.float32(np.inf)))

    specials_bits = np.array(
        [
            0x00000000, 0x80000000,              # +0, -0
            0x00000001, 0x80000001,              # smallest f32 subnormals
            0x007FFFFF, 0x807FFFFF,              # largest f32 subnormals
            0x00800000, 0x80800000,              # smallest f32 normals
            0x7F7FFFFF, 0xFF7FFFFF,              # +/- f32 max
            0x7F800000, 0xFF800000,              # +/- inf
            0x7FC00000, 0xFFC00000,              # quiet NaN
            0x7FA00000, 0xFFA00000,              # signaling NaN
            0x7F800001, 0xFF800001,              # minimal-payload NaN
            0x3F800000, 0xBF800000,              # +/- 1.0
        ],
        dtype="<u4",
    )
    parts.append(specials_bits.view(np.float32))

    rng = np.random.default_rng(RNG_SEED)
    span = float(maxfin) * 1.5
    parts.append(rng.uniform(-span, span, N_RANDOM_LINEAR).astype(np.float32))
    tiny = float(np.min(np.abs(finite[finite != 0.0])))
    mags = np.exp(rng.uniform(np.log(tiny * 0.25), np.log(span), N_RANDOM_LOG))
    signs = rng.choice(np.array([-1.0, 1.0]), N_RANDOM_LOG)
    parts.append((mags * signs).astype(np.float32))

    allbits = np.concatenate([np_f32_bits(p) for p in parts])
    return np.unique(allbits)


# ---------------------------------------------------------------------------
# section builders
# ---------------------------------------------------------------------------

ARITH = (("f8_add", OP_ADD), ("f8_sub", OP_SUB), ("f8_mul", OP_MUL),
         ("f8_div", OP_DIV))


def f8_is_nan(bits, fmt):
    u = bits.astype("<u4") & 0xFF
    if fmt == ORACLE.E4M3:
        return (u & 0x7F) == 0x7F
    return ((u & 0x7C) == 0x7C) & ((u & 0x03) != 0)


def crosscheck(label, fmt, torch_out, oracle_out, report):
    """torch and the oracle must agree bit for bit on RNE values, except where
    both say NaN: torch keeps sign and payload, softfloat canonicalizes."""
    diff = torch_out != oracle_out
    if not diff.any():
        return
    both_nan = f8_is_nan(torch_out, fmt) & f8_is_nan(oracle_out, fmt)
    real = diff & ~both_nan
    if real.any():
        idx = np.flatnonzero(real)
        report.append(f"{label}: {len(idx)} value disagreements, first at "
                      f"index {idx[0]} torch=0x{torch_out[idx[0]]:02x} "
                      f"oracle=0x{oracle_out[idx[0]]:02x}")


def build_sections(name, fmt, dt, report):
    sections = []
    idx = np.arange(256, dtype="<u4")
    wide = torch.arange(256, dtype=torch.uint8).view(dt).to(torch.float32)

    sections.append(
        (f"{name}/f8_to_f32", OP_WIDEN_F32, fmt,
         make_records(idx, None, None, f32_bits(wide), OP_WIDEN_F32, fmt))
    )

    a = wide.repeat_interleave(256)
    b = wide.repeat(256)
    ai = np.repeat(idx, 256)
    bi = np.tile(idx, 256)

    for opname, op in ARITH:
        fn = {OP_ADD: torch.add, OP_SUB: torch.sub, OP_MUL: torch.mul,
              OP_DIV: torch.div}[op]
        out = f8_bits(fn(a, b).to(dt))
        crosscheck(f"{name}/{opname}", fmt, out,
                   oracle_values(op, fmt, ai, bi, ORACLE.RNE), report)
        sections.append((f"{name}/{opname}", op, fmt,
                         make_records(ai, bi, None, out, op, fmt)))

    for opname, op, fn in (
        ("f8_eq", OP_EQ, torch.eq),
        ("f8_le", OP_LE, torch.le),
        ("f8_lt", OP_LT, torch.lt),
    ):
        out = fn(a, b).numpy().astype("<u4")
        sections.append((f"{name}/{opname}", op, fmt,
                         make_records(ai, bi, None, out, op, fmt)))

    nbits = narrow_inputs(dt)
    nvals = torch.from_numpy(nbits.view(np.float32).copy())
    nout = f8_bits(nvals.to(dt))
    crosscheck(f"{name}/f32_to_f8", fmt, nout,
               oracle_values(OP_NARROW_F8, fmt, nbits, None, ORACLE.RNE),
               report)
    sections.append((f"{name}/f32_to_f8", OP_NARROW_F8, fmt,
                     make_records(nbits, None, None, nout, OP_NARROW_F8, fmt)))

    # the directed and ties-away modes have no torch reference, so value and
    # flags both come from the oracle
    for mode in ORACLE.RMODES:
        if mode == ORACLE.RNE:
            continue
        tag = ORACLE.RMODE_TAG[mode]
        for opname, op in ARITH:
            out = oracle_values(op, fmt, ai, bi, mode)
            sections.append((f"{name}/{opname}@{tag}", op, fmt,
                             make_records(ai, bi, None, out, op, fmt, mode)))
        out = oracle_values(OP_NARROW_F8, fmt, nbits, None, mode)
        sections.append(
            (f"{name}/f32_to_f8@{tag}", OP_NARROW_F8, fmt,
             make_records(nbits, None, None, out, OP_NARROW_F8, fmt, mode)))

    return sections


def write_golden(path, sections):
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(sections)))
        for name, op, fmt, rec in sections:
            raw = name.encode("ascii")
            if len(raw) >= NAME_LEN:
                raise ValueError(f"section name too long: {name}")
            fh.write(raw.ljust(NAME_LEN, b"\0"))
            fh.write(struct.pack("<IIII", op, fmt, len(rec), REC_LEN))
            fh.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser(description="generate the fp8 golden file")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    out = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "golden.bin")

    for _, _, attr in FORMATS:
        if not hasattr(torch, attr):
            sys.exit(f"torch {torch.__version__} lacks torch.{attr}")

    sections = []
    report = []
    for name, fmt, attr in FORMATS:
        sections += build_sections(name, fmt, getattr(torch, attr), report)

    write_golden(out, sections)

    if not args.quiet:
        print(f"torch {torch.__version__}")
        print(f"golden {out} ({os.path.getsize(out)} bytes)")
        for name, _, _, rec in sections:
            known = int(rec["known"].sum())
            print(f"  {name:<24} {len(rec)} cases, {known} flag goldens")

    if report:
        print("\noracle-vs-torch value disagreements (RNE):")
        for line in report:
            print(f"  {line}")
    else:
        print("\noracle agrees with torch on every RNE value"
              " (NaN encodings aside)")


if __name__ == "__main__":
    main()
