#!/usr/bin/env python3
"""Second opinion on the spike fp8 vector stream, independent of softfloat.
ref.c links the same library spike does, so it can only prove the wiring; this
recomputes every answer from exact rationals via tests/fp8/fp8_oracle.py."""

import argparse
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fp8"))

import fp8_oracle as O  # noqa: E402

MAGIC = b"FP8VEC\x01\x00"
END = b"ENDFP8VEC\x00\x00\x00"
NAME_LEN = 32
HDR_LEN = 44

NX, UF, OF, DZ, NV = O.NX, O.UF, O.OF, O.DZ, O.NV
E4M3 = O.E4M3

FRM2MODE = {0: O.RNE, 1: O.RTZ, 2: O.RDN, 3: O.RUP, 4: O.RMM}


# --------------------------------------------------------------------------
# f16, reusing the oracle's exact-rational machinery
# --------------------------------------------------------------------------

F16_P = 11
F16_EMIN = -14
F16_MAXFIN = (2047, 5)           # 2047 * 32 = 65504
F16_BIAS = 15
F16_MANT = 10


def decode_f16(ui):
    ui &= 0xFFFF
    sign = bool(ui >> 15)
    exp = (ui >> 10) & 0x1F
    frac = ui & 0x3FF
    if exp == 0x1F:
        if not frac:
            return O.Val("inf", sign)
        return O.Val("qnan" if frac & 0x200 else "snan", sign)
    if exp == 0:
        return O.Val("num", sign, *O._mag(frac, -24))
    return O.Val("num", sign, *O._mag(frac | 0x400, exp - 25))


def round_to_f16(sign, num, den, mode):
    """Same algorithm as fp8_oracle.round_to_f8, retargeted to binary16."""
    sbit = 0x8000 if sign else 0
    if num == 0:
        return sbit, 0
    e = O._floor_log2(num, den)
    q_ub = e - F16_P + 1
    n_ub, _ = O._round_scaled(num, den, q_ub, mode, sign)
    tiny = O._lt_pow2(n_ub, q_ub, F16_EMIN)
    q = max(e, F16_EMIN) - F16_P + 1
    n, inexact = O._round_scaled(num, den, q, mode, sign)
    if O._gt_dyadic(n, q, *F16_MAXFIN):
        away = O._overflow_goes_to_inf(mode, sign)
        return sbit | (0x7C00 if away else 0x7BFF), OF | NX
    flags = NX if inexact else 0
    if tiny and inexact:
        flags |= UF
    if n >= 1 << F16_P:
        n >>= 1
        q += 1
    if n >= 1 << F16_MANT:
        return sbit | ((q + F16_MANT + F16_BIAS) << F16_MANT) | \
            (n - (1 << F16_MANT)), flags
    return sbit | n, flags


def f16_nan_prologue(*vals):
    if any(v.kind == "snan" for v in vals):
        return 0x7E00, NV
    if any(v.kind == "qnan" for v in vals):
        return 0x7E00, 0
    return None


def f16_addsub(a, b, mode, negate_b):
    sb = (not b.sign) if negate_b else b.sign
    pre = f16_nan_prologue(a, b)
    if pre:
        return pre
    if a.kind == "inf" or b.kind == "inf":
        if a.kind == "inf" and b.kind == "inf":
            if a.sign != sb:
                return 0x7E00, NV
            return (0x8000 if a.sign else 0) | 0x7C00, 0
        s = a.sign if a.kind == "inf" else sb
        return (0x8000 if s else 0) | 0x7C00, 0
    num = (-a.num if a.sign else a.num) * b.den \
        + (-b.num if sb else b.num) * a.den
    den = a.den * b.den
    if num == 0:
        sign = a.sign if a.sign == sb else (mode == O.RDN)
        return (0x8000 if sign else 0), 0
    return round_to_f16(num < 0, abs(num), den, mode)


def f16_mul_vals(a, b, mode):
    pre = f16_nan_prologue(a, b)
    if pre:
        return pre
    sign = a.sign != b.sign
    if a.kind == "inf" or b.kind == "inf":
        if a.is_zero or b.is_zero:
            return 0x7E00, NV
        return (0x8000 if sign else 0) | 0x7C00, 0
    return round_to_f16(sign, a.num * b.num, a.den * b.den, mode)


# --------------------------------------------------------------------------
# fused multiply-add, exact in both widths
# --------------------------------------------------------------------------

def _fma_exact(a, b, c, mode, nan_ui, inf_ui, sbit, rounder):
    """a*b+c rounded once.  Returns (bits, flags)."""
    if a.kind == "snan" or b.kind == "snan" or c.kind == "snan":
        return nan_ui, NV
    prod_inf = (a.kind == "inf" or b.kind == "inf")
    if prod_inf and (a.is_zero or b.is_zero):
        return nan_ui, NV
    if a.is_nan or b.is_nan or c.is_nan:
        return nan_ui, 0
    psign = a.sign != b.sign
    if prod_inf:
        if c.kind == "inf" and c.sign != psign:
            return nan_ui, NV
        return (sbit if psign else 0) | inf_ui, 0
    if c.kind == "inf":
        return (sbit if c.sign else 0) | inf_ui, 0

    pnum, pden = a.num * b.num, a.den * b.den
    num = (-pnum if psign else pnum) * c.den \
        + (-c.num if c.sign else c.num) * pden
    den = pden * c.den
    if num == 0:
        pzero = a.is_zero or b.is_zero
        if pzero and c.is_zero:
            sign = psign if psign == c.sign else (mode == O.RDN)
        elif pzero:
            sign = c.sign
        else:
            sign = (mode == O.RDN)
        return (sbit if sign else 0), 0
    return rounder(num < 0, abs(num), den, mode)


def fma8(ua, ub, uc, mode, negp=False, negc=False):
    a = O.decode_f8(ua ^ (0x80 if negp else 0), E4M3)
    b = O.decode_f8(ub, E4M3)
    c = O.decode_f8(uc ^ (0x80 if negc else 0), E4M3)
    f = O.FORMATS[E4M3]
    return _fma_exact(a, b, c, mode, f.nan_ui, f.inf_ui, 0x80,
                      lambda s, n, d, m: O.round_to_f8(s, n, d, E4M3, m))


def fma16(a16, b16, c16, mode):
    return _fma_exact(a16, b16, c16, mode, 0x7E00, 0x7C00, 0x8000,
                      round_to_f16)


# --------------------------------------------------------------------------
# per-instruction expectations
# --------------------------------------------------------------------------


def op_min_max(ua, ub, mode, want_max):
    a, b = O.decode_f8(ua, E4M3), O.decode_f8(ub, E4M3)
    nan_ui = O.FORMATS[E4M3].nan_ui
    if a.is_nan and b.is_nan:
        return nan_ui, (NV if (a.kind == "snan" or b.kind == "snan") else 0)
    fl = NV if (a.kind == "snan" or b.kind == "snan") else 0
    if a.is_nan:
        return ub, fl
    if b.is_nan:
        return ua, fl
    lt = _lt_f8(a, b)
    gt = _lt_f8(b, a)
    if not lt and not gt:
        # equal magnitudes including +-0: min takes the negative one
        if a.is_zero and b.is_zero and a.sign != b.sign:
            neg = ua if a.sign else ub
            pos = ub if a.sign else ua
            return (pos if want_max else neg), fl
        return ua, fl
    if want_max:
        return (ua if gt else ub), fl
    return (ua if lt else ub), fl


def _lt_f8(a, b):
    if a.kind == "inf" and b.kind == "inf":
        return a.sign and not b.sign
    if a.kind == "inf":
        return a.sign
    if b.kind == "inf":
        return not b.sign
    av = (-a.num, a.den) if a.sign else (a.num, a.den)
    bv = (-b.num, b.den) if b.sign else (b.num, b.den)
    return av[0] * bv[1] < bv[0] * av[1]


def cmp_pred(ua, ub, kind):
    a, b = O.decode_f8(ua, E4M3), O.decode_f8(ub, E4M3)
    unordered = a.is_nan or b.is_nan
    if kind == "eq":
        flags = O.cmp_flags(ua, ub, E4M3, False)
        if unordered:
            return 0, flags
        return int(not _lt_f8(a, b) and not _lt_f8(b, a)), flags
    if kind == "ne":
        flags = O.cmp_flags(ua, ub, E4M3, False)
        if unordered:
            return 1, flags
        return int(_lt_f8(a, b) or _lt_f8(b, a)), flags
    flags = O.cmp_flags(ua, ub, E4M3, True)
    if unordered:
        return 0, flags
    if kind == "lt":
        return int(_lt_f8(a, b)), flags
    return int(not _lt_f8(b, a)), flags


def op_sqrt(ua, mode):
    a = O.decode_f8(ua, E4M3)
    f = O.FORMATS[E4M3]
    if a.kind == "snan":
        return f.nan_ui, NV
    if a.is_nan:
        return f.nan_ui, 0
    if a.is_zero:
        return ua, 0
    if a.sign:
        return f.nan_ui, NV
    if a.kind == "inf":
        return f.inf_ui, 0
    # correctly rounded sqrt: pick the fp8 neighbour by comparing squares
    return _round_sqrt(a.num, a.den, mode)


def _isqrt(x):
    r = int(x ** 0.5)
    while r * r > x:
        r -= 1
    while (r + 1) * (r + 1) <= x:
        r += 1
    return r


def _cmp_scaled_sq(n, q, num, den):
    """sign of (n*2**q)**2 - num/den, as an integer comparison."""
    if q >= 0:
        lhs, rhs = n * n * (1 << (2 * q)) * den, num
    else:
        lhs, rhs = n * n * den, num * (1 << (-2 * q))
    return (lhs > rhs) - (lhs < rhs)


def _round_sqrt(num, den, mode):
    """Correctly round sqrt(num/den) to fp8; integer arithmetic only."""
    f = O.FORMATS[E4M3]
    e = O._floor_log2(num, den) // 2
    while _cmp_scaled_sq(1, e + 1, num, den) <= 0:
        e += 1
    while _cmp_scaled_sq(1, e, num, den) > 0:
        e -= 1

    q = max(e, f.emin) - f.p + 1
    # floor of the root measured in units of 2**q
    if q <= 0:
        n = _isqrt((num << (-2 * q)) // den)
    else:
        n = _isqrt(num // (den << (2 * q)))
    while _cmp_scaled_sq(n + 1, q, num, den) <= 0:
        n += 1

    if _cmp_scaled_sq(n, q, num, den) == 0:
        cand, inexact = n, False
    else:
        cand, inexact = _pick_sqrt(n, q, num, den, mode), True

    if O._gt_dyadic(cand, q, *f.maxfin):
        away = O._overflow_goes_to_inf(mode, False)
        return (f.inf_ui if away else f.maxfin_ui), OF | NX
    flags = NX if inexact else 0
    if inexact and O._lt_pow2(cand, q, f.emin):
        flags |= UF
    if cand >= 1 << f.p:
        cand >>= 1
        q += 1
    if cand >= 1 << f.mant:
        return ((q + f.mant + f.bias) << f.mant) | (cand - (1 << f.mant)), flags
    return cand, flags


def _pick_sqrt(n, q, num, den, mode):
    """Choose n or n+1 at scale 2**q; the root is strictly between them."""
    if mode in (O.RTZ, O.RDN):
        return n
    if mode == O.RUP:
        return n + 1
    # nearest: the midpoint is (2n+1)*2**(q-1); a square root is never exactly
    # a midpoint of a binary grid, so no tie can occur
    return n + 1 if _cmp_scaled_sq(2 * n + 1, q - 1, num, den) < 0 else n


# --------------------------------------------------------------------------
# stream walking
# --------------------------------------------------------------------------

def rd(buf, off, w):
    v = 0
    for i in range(w):
        v |= buf[off + i] << (8 * i)
    return v


def sections(path):
    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:8] == MAGIC, "bad magic"
    off = 8
    while off + HDR_LEN <= len(data):
        if data[off:off + len(END)] == END:
            return
        name = data[off:off + NAME_LEN].split(b"\0")[0].decode()
        n_in, w0, w1, w2, out_w, frm = struct.unpack_from("<BBBBBB", data,
                                                          off + NAME_LEN)
        count = struct.unpack_from("<I", data, off + 40)[0]
        off += HDR_LEN
        rw = w0 + w1 + w2 + out_w + 1
        yield name, (n_in, (w0, w1, w2), out_w, frm), data[off:off + rw * count], rw, count
        off += rw * count


def base_name(name):
    return name.split("@")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap cases per section (0 = all)")
    ap.add_argument("--max-report", type=int, default=6)
    ap.add_argument("--mutate", type=int, default=0)
    args = ap.parse_args()

    total = bad = fbad = 0
    checked_sections = 0
    rows = []

    for name, meta, blob, rw, count in sections(args.stream):
        n_in, widths, out_w, frm = meta
        fn = EXPECT.get(base_name(name))
        if fn is None:
            rows.append((name, count, "-", "-", "skip"))
            continue
        mode = FRM2MODE[frm]
        n = count if not args.limit else min(count, args.limit)
        # the LMUL=8 sections put the whole group's OR in lane 0, so a
        # per-case flag compare does not apply to them
        chunked = name.startswith("lmul8.")
        chunk_want = chunk_got = 0
        v_bad = f_bad = 0
        reported = 0
        for k in range(n):
            off = rw * k
            ins = []
            p = off
            for i in range(3):
                if not widths[i]:
                    break
                ins.append(rd(blob, p, widths[i]))
                p += widths[i]
            got = rd(blob, p, out_w)
            gfl = blob[p + out_w]
            want, wfl = fn(ins, mode)
            if args.mutate and (k % args.mutate) == 0:
                want ^= 1
            if chunked:
                if (k & 127) == 0:
                    chunk_got, chunk_want = gfl, 0
                chunk_want |= wfl
                gfl = wfl
                if (k & 127) == 127 and chunk_got != chunk_want:
                    f_bad += 1
            if got != want or gfl != wfl:
                if got != want:
                    v_bad += 1
                if gfl != wfl:
                    f_bad += 1
                if reported < args.max_report:
                    reported += 1
                    print("ORACLE %-20s frm=%d in=%s got=0x%x/0x%02x "
                          "want=0x%x/0x%02x" % (name, frm,
                                                [hex(x) for x in ins],
                                                got, gfl, want, wfl),
                          file=sys.stderr)
        total += n
        bad += v_bad
        fbad += f_bad
        checked_sections += 1
        rows.append((name, n, v_bad, f_bad,
                     "ok" if not (v_bad or f_bad) else "FAIL"))

    print("%-24s %10s %10s %10s %6s" % ("section", "cases", "val_bad",
                                        "flag_bad", "state"))
    for r in rows:
        print("%-24s %10s %10s %10s %6s" % r)
    print("\nTOTAL sections_checked=%d cases=%d value_mismatch=%d "
          "fflags_mismatch=%d" % (checked_sections, total, bad, fbad))
    print("RESULT: %s" % ("FAIL" if (bad or fbad) else "PASS"))
    return 1 if (bad or fbad) else 0


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------

def _bin(f):
    return lambda ins, mode: f(ins[0], ins[1], mode)


EXPECT = {
    "vfadd.vv": _bin(lambda a, b, m: O.op_add(a, b, E4M3, m)),
    "vfadd.vf": _bin(lambda a, b, m: O.op_add(a, b, E4M3, m)),
    "vfsub.vv": _bin(lambda a, b, m: O.op_sub(a, b, E4M3, m)),
    "vfsub.vf": _bin(lambda a, b, m: O.op_sub(a, b, E4M3, m)),
    "vfrsub.vf": _bin(lambda a, b, m: O.op_sub(b, a, E4M3, m)),
    "vfmul.vv": _bin(lambda a, b, m: O.op_mul(a, b, E4M3, m)),
    "vfmul.vf": _bin(lambda a, b, m: O.op_mul(a, b, E4M3, m)),
    "vfdiv.vv": _bin(lambda a, b, m: O.op_div(a, b, E4M3, m)),
    "vfdiv.vf": _bin(lambda a, b, m: O.op_div(a, b, E4M3, m)),
    "vfrdiv.vf": _bin(lambda a, b, m: O.op_div(b, a, E4M3, m)),
    "vfmin.vv": _bin(lambda a, b, m: op_min_max(a, b, m, False)),
    "vfmin.vf": _bin(lambda a, b, m: op_min_max(a, b, m, False)),
    "vfmax.vv": _bin(lambda a, b, m: op_min_max(a, b, m, True)),
    "vfmax.vf": _bin(lambda a, b, m: op_min_max(a, b, m, True)),
    "vfsgnj.vv": _bin(lambda a, b, m: ((a & 0x7F) | (b & 0x80), 0)),
    "vfsgnj.vf": _bin(lambda a, b, m: ((a & 0x7F) | (b & 0x80), 0)),
    "vfsgnjn.vv": _bin(lambda a, b, m: ((a & 0x7F) | ((b ^ 0x80) & 0x80), 0)),
    "vfsgnjn.vf": _bin(lambda a, b, m: ((a & 0x7F) | ((b ^ 0x80) & 0x80), 0)),
    "vfsgnjx.vv": _bin(lambda a, b, m: ((a & 0x7F) | ((a ^ b) & 0x80), 0)),
    "vfsgnjx.vf": _bin(lambda a, b, m: ((a & 0x7F) | ((a ^ b) & 0x80), 0)),
    "vmfeq.vv": _bin(lambda a, b, m: cmp_pred(a, b, "eq")),
    "vmfeq.vf": _bin(lambda a, b, m: cmp_pred(a, b, "eq")),
    "vmfne.vv": _bin(lambda a, b, m: cmp_pred(a, b, "ne")),
    "vmfne.vf": _bin(lambda a, b, m: cmp_pred(a, b, "ne")),
    "vmflt.vv": _bin(lambda a, b, m: cmp_pred(a, b, "lt")),
    "vmflt.vf": _bin(lambda a, b, m: cmp_pred(a, b, "lt")),
    "vmfle.vv": _bin(lambda a, b, m: cmp_pred(a, b, "le")),
    "vmfle.vf": _bin(lambda a, b, m: cmp_pred(a, b, "le")),
    "vmfgt.vf": _bin(lambda a, b, m: cmp_pred(b, a, "lt")),
    "vmfge.vf": _bin(lambda a, b, m: cmp_pred(b, a, "le")),
    "vfsqrt.v": lambda ins, m: op_sqrt(ins[0], m),
    "lmul8.vfadd.vv": _bin(lambda a, b, m: O.op_add(a, b, E4M3, m)),
    "lmul8.vfmul.vv": _bin(lambda a, b, m: O.op_mul(a, b, E4M3, m)),
}

# a=vs2, b=vs1, c=vd; the eight fused forms differ only in which signs flip
_FMA = {
    "vfmacc": (False, False, lambda a, b, c: (b, a, c)),
    "vfnmacc": (True, True, lambda a, b, c: (b, a, c)),
    "vfmsac": (False, True, lambda a, b, c: (b, a, c)),
    "vfnmsac": (True, False, lambda a, b, c: (b, a, c)),
    "vfmadd": (False, False, lambda a, b, c: (c, b, a)),
    "vfnmadd": (True, True, lambda a, b, c: (c, b, a)),
    "vfmsub": (False, True, lambda a, b, c: (c, b, a)),
    "vfnmsub": (True, False, lambda a, b, c: (c, b, a)),
}


def _mk_fma(negp, negc, order):
    def f(ins, mode):
        x, y, z = order(ins[0], ins[1], ins[2])
        return fma8(x, y, z, mode, negp, negc)
    return f


for _stem, (_np, _nc, _ord) in _FMA.items():
    EXPECT[_stem + ".vv"] = _mk_fma(_np, _nc, _ord)
    EXPECT[_stem + ".vf"] = _mk_fma(_np, _nc, _ord)
EXPECT["lmul8.vfmacc.vv"] = _mk_fma(False, False, _FMA["vfmacc"][2])


def _w(u):
    return O.decode_f8(u, E4M3)


def _w16_of_f8(u):
    v = _w(u)
    if v.kind == "snan":
        return O.Val("snan", v.sign)
    if v.kind == "qnan":
        return O.Val("qnan", v.sign)
    return v


EXPECT["vfwadd.vv"] = lambda ins, m: f16_addsub(_w16_of_f8(ins[0]),
                                                _w16_of_f8(ins[1]), m, False)
EXPECT["vfwadd.vf"] = EXPECT["vfwadd.vv"]
EXPECT["vfwsub.vv"] = lambda ins, m: f16_addsub(_w16_of_f8(ins[0]),
                                                _w16_of_f8(ins[1]), m, True)
EXPECT["vfwsub.vf"] = EXPECT["vfwsub.vv"]
EXPECT["vfwmul.vv"] = lambda ins, m: f16_mul_vals(_w16_of_f8(ins[0]),
                                                  _w16_of_f8(ins[1]), m)
EXPECT["vfwmul.vf"] = EXPECT["vfwmul.vv"]
EXPECT["vfwadd.wv"] = lambda ins, m: f16_addsub(decode_f16(ins[0]),
                                                _w16_of_f8(ins[1]), m, False)
EXPECT["vfwadd.wf"] = EXPECT["vfwadd.wv"]
EXPECT["vfwsub.wv"] = lambda ins, m: f16_addsub(decode_f16(ins[0]),
                                                _w16_of_f8(ins[1]), m, True)
EXPECT["vfwsub.wf"] = EXPECT["vfwsub.wv"]


def _neg(v):
    if v.kind in ("inf", "num"):
        return O.Val(v.kind, not v.sign, v.num, v.den)
    return O.Val(v.kind, not v.sign)


def _mk_wfma(negp, negc):
    def f(ins, mode):
        a = _w16_of_f8(ins[0])
        b = _w16_of_f8(ins[1])
        c = decode_f16(ins[2])
        if negp:
            b = _neg(b)
        if negc:
            c = _neg(c)
        return fma16(b, a, c, mode)
    return f


for _stem, _sg in (("vfwmacc", (False, False)), ("vfwnmacc", (True, True)),
                   ("vfwmsac", (False, True)), ("vfwnmsac", (True, False))):
    EXPECT[_stem + ".vv"] = _mk_wfma(*_sg)
    EXPECT[_stem + ".vf"] = _mk_wfma(*_sg)


def _red(op):
    def f(ins, mode):
        acc = ins[1]
        flags = 0
        for k in range(8):
            acc, fl = op(acc, (ins[0] >> (8 * k)) & 0xFF, mode)
            flags |= fl
        return acc, flags
    return f


EXPECT["vfredosum.vs"] = _red(lambda a, b, m: O.op_add(a, b, E4M3, m))
EXPECT["vfredusum.vs"] = EXPECT["vfredosum.vs"]
EXPECT["vfredmin.vs"] = _red(lambda a, b, m: op_min_max(a, b, m, False))
EXPECT["vfredmax.vs"] = _red(lambda a, b, m: op_min_max(a, b, m, True))


def _wred(ins, mode):
    bits = ins[1]
    flags = 0
    for k in range(8):
        bits, fl = f16_addsub(decode_f16(bits),
                              _w16_of_f8((ins[0] >> (8 * k)) & 0xFF),
                              mode, False)
        flags |= fl
    return bits, flags


EXPECT["vfwredosum.vs"] = _wred
EXPECT["vfwredusum.vs"] = _wred

EXPECT["vfncvt.f.f.w"] = lambda ins, m: _narrow16(ins[0], m)
EXPECT["vfncvt.rod.f.f.w"] = lambda ins, m: _narrow16_rod(ins[0])


def _narrow16(u16, mode):
    a = decode_f16(u16)
    f = O.FORMATS[E4M3]
    if a.kind == "snan":
        return f.nan_ui, NV
    if a.is_nan:
        return f.nan_ui, 0
    if a.kind == "inf":
        return (0x80 if a.sign else 0) | f.inf_ui, 0
    return O.round_to_f8(a.sign, a.num, a.den, E4M3, mode)


def _narrow16_rod(u16):
    """Round to odd: never lands on an even significand unless exact."""
    a = decode_f16(u16)
    f = O.FORMATS[E4M3]
    if a.kind == "snan":
        return f.nan_ui, NV
    if a.is_nan:
        return f.nan_ui, 0
    if a.kind == "inf":
        return (0x80 if a.sign else 0) | f.inf_ui, 0
    down, fl_d = O.round_to_f8(a.sign, a.num, a.den, E4M3, O.RTZ)
    if not (fl_d & NX):
        return down, fl_d
    if down & 1:
        return down, fl_d
    up, fl_u = O.round_to_f8(a.sign, a.num, a.den, E4M3,
                             O.RUP if not a.sign else O.RDN)
    # E4M3's largest finite has an even significand, so the odd neighbour
    # above it does not exist; that is an overflow, not a NaN
    if fl_u & OF:
        return (0x80 if a.sign else 0) | f.maxfin_ui, OF | NX
    return up, fl_u | NX


def _int_to_f8(v, mode):
    if v == 0:
        return 0, 0
    return O.round_to_f8(v < 0, abs(v), 1, E4M3, mode)


EXPECT["vfncvt.f.x.w"] = lambda ins, m: _int_to_f8(
    ins[0] - 0x10000 if ins[0] >= 0x8000 else ins[0], m)
EXPECT["vfncvt.f.xu.w"] = lambda ins, m: _int_to_f8(ins[0], m)
EXPECT["vfcvt.f.x.v"] = lambda ins, m: _int_to_f8(
    ins[0] - 0x100 if ins[0] >= 0x80 else ins[0], m)
EXPECT["vfcvt.f.xu.v"] = lambda ins, m: _int_to_f8(ins[0], m)


def _f8_to_int(u, mode, bits, signed, rtz):
    a = O.decode_f8(u, E4M3)
    lo = -(1 << (bits - 1)) if signed else 0
    hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    mask = (1 << bits) - 1
    if a.is_nan:
        return (hi if not signed else hi) & mask, NV
    if a.kind == "inf":
        return (hi if not a.sign else lo) & mask, NV
    m = O.RTZ if rtz else mode
    n, inexact = O._round_scaled(a.num, a.den, 0, m, a.sign)
    val = -n if a.sign else n
    if val < lo or val > hi:
        return (lo if val < lo else hi) & mask, NV
    return val & mask, (NX if inexact else 0)


EXPECT["vfcvt.x.f.v"] = lambda ins, m: _f8_to_int(ins[0], m, 8, True, False)
EXPECT["vfcvt.xu.f.v"] = lambda ins, m: _f8_to_int(ins[0], m, 8, False, False)
EXPECT["vfcvt.rtz.x.f.v"] = lambda ins, m: _f8_to_int(ins[0], m, 8, True, True)
EXPECT["vfcvt.rtz.xu.f.v"] = lambda ins, m: _f8_to_int(ins[0], m, 8, False, True)
EXPECT["vfwcvt.x.f.v"] = lambda ins, m: _f8_to_int(ins[0], m, 16, True, False)
EXPECT["vfwcvt.xu.f.v"] = lambda ins, m: _f8_to_int(ins[0], m, 16, False, False)
EXPECT["vfwcvt.rtz.x.f.v"] = lambda ins, m: _f8_to_int(ins[0], m, 16, True, True)
EXPECT["vfwcvt.rtz.xu.f.v"] = lambda ins, m: _f8_to_int(ins[0], m, 16, False,
                                                        True)


def _widen_f8_to_f16(u):
    a = O.decode_f8(u, E4M3)
    if a.is_nan:
        return 0x7E00, (NV if a.kind == "snan" else 0)
    if a.kind == "inf":
        return (0x8000 if a.sign else 0) | 0x7C00, 0
    if a.num == 0:
        return (0x8000 if a.sign else 0), 0
    return round_to_f16(a.sign, a.num, a.den, O.RNE)


EXPECT["vfwcvt.f.f.v"] = lambda ins, m: _widen_f8_to_f16(ins[0])


if __name__ == "__main__":
    sys.exit(main())
