#!/usr/bin/env python3
"""Exact fp8 oracle: values and IEEE exception flags from first principles.
Operands are exact rationals and every decision is an integer comparison, so
no floating point takes part and the f32-and-narrow path is checked, not echoed.
"""

# softfloat_exceptionFlags bit values
NX = 1
UF = 2
OF = 4
DZ = 8
NV = 16

# softfloat_roundingMode values
RNE = 0
RTZ = 1
RDN = 2
RUP = 3
RMM = 4

RMODES = (RNE, RTZ, RDN, RUP, RMM)
RMODE_TAG = {RNE: "rne", RTZ: "rtz", RDN: "rdn", RUP: "rup", RMM: "rmm"}

E4M3 = 0
E5M2 = 1


class Format:
    """OCP fp8 parameters.  'maxfin' is (n, q) for the largest finite, n*2**q;
    E4M3 stops at 448 because S.1111.111 is spent on its only NaN."""

    def __init__(self, code, mant, bias, emin, maxfin, nan_ui, inf_ui,
                 maxfin_ui, has_inf):
        self.code = code
        self.mant = mant
        self.p = mant + 1
        self.bias = bias
        self.emin = emin
        self.maxfin = maxfin
        self.nan_ui = nan_ui
        self.inf_ui = inf_ui
        self.maxfin_ui = maxfin_ui
        self.has_inf = has_inf


FORMATS = {
    E4M3: Format(E4M3, mant=3, bias=7, emin=-6, maxfin=(14, 5), nan_ui=0x7F,
                 inf_ui=0x7F, maxfin_ui=0x7E, has_inf=False),
    E5M2: Format(E5M2, mant=2, bias=15, emin=-14, maxfin=(7, 13), nan_ui=0x7E,
                 inf_ui=0x7C, maxfin_ui=0x7B, has_inf=True),
}


# ---------------------------------------------------------------------------
# integer helpers on dyadic and rational magnitudes
# ---------------------------------------------------------------------------

def _ge_pow2(num, den, e):
    """num/den >= 2**e, with num > 0 and den > 0."""
    if e >= 0:
        return num >= den << e
    return num << -e >= den


def _floor_log2(num, den):
    e = num.bit_length() - den.bit_length()
    while not _ge_pow2(num, den, e):
        e -= 1
    while _ge_pow2(num, den, e + 1):
        e += 1
    return e


def _round_scaled(num, den, q, mode, sign):
    """Round num/den to an integer multiple of 2**q; returns (n, inexact)."""
    if q <= 0:
        n_num, n_den = num << -q, den
    else:
        n_num, n_den = num, den << q
    n, r = divmod(n_num, n_den)
    if not r:
        return n, False
    if mode == RNE:
        twice = r << 1
        if twice > n_den or (twice == n_den and (n & 1)):
            n += 1
    elif mode == RMM:
        if (r << 1) >= n_den:
            n += 1
    elif mode == RUP:
        if not sign:
            n += 1
    elif mode == RDN:
        if sign:
            n += 1
    return n, True


def _lt_pow2(n, q, e):
    """n*2**q < 2**e, with n >= 0."""
    if q <= e:
        return n < 1 << (e - q)
    return False


def _gt_dyadic(n, q, mn, mq):
    """n*2**q > mn*2**mq."""
    if q >= mq:
        return n << (q - mq) > mn
    return n > mn << (mq - q)


def _overflow_goes_to_inf(mode, sign):
    return (mode in (RNE, RMM) or (mode == RUP and not sign)
            or (mode == RDN and sign))


# ---------------------------------------------------------------------------
# the one narrowing step: exact magnitude -> fp8 bits plus flags
# ---------------------------------------------------------------------------

_round_cache = {}


def round_to_f8(sign, num, den, fmt_code, mode, tininess_after=True):
    """Correctly round the exact rational sign*num/den to fp8.
    Returns (bits, flags); flags carry only NX, UF and OF."""
    key = (sign, num, den, fmt_code, mode, tininess_after)
    hit = _round_cache.get(key)
    if hit is not None:
        return hit
    out = _round_to_f8(sign, num, den, fmt_code, mode, tininess_after)
    _round_cache[key] = out
    return out


def _round_to_f8(sign, num, den, fmt_code, mode, tininess_after):
    f = FORMATS[fmt_code]
    sbit = 0x80 if sign else 0
    if num == 0:
        return sbit, 0

    e = _floor_log2(num, den)
    p = f.p

    # tininess is judged on the result rounded with unbounded exponent range
    q_ub = e - p + 1
    n_ub, _ = _round_scaled(num, den, q_ub, mode, sign)
    if tininess_after:
        tiny = _lt_pow2(n_ub, q_ub, f.emin)
    else:
        tiny = not _ge_pow2(num, den, f.emin)

    q = max(e, f.emin) - p + 1
    n, inexact = _round_scaled(num, den, q, mode, sign)

    if _gt_dyadic(n, q, *f.maxfin):
        away = _overflow_goes_to_inf(mode, sign)
        return sbit | (f.inf_ui if away else f.maxfin_ui), OF | NX

    flags = NX if inexact else 0
    if tiny and inexact:
        flags |= UF

    if n >= 1 << p:
        n >>= 1
        q += 1
    if n >= 1 << f.mant:
        exp_field = q + f.mant + f.bias
        return sbit | (exp_field << f.mant) | (n - (1 << f.mant)), flags
    return sbit | n, flags


# ---------------------------------------------------------------------------
# fp8 and f32 decoding
# ---------------------------------------------------------------------------

class Val:
    __slots__ = ("kind", "sign", "num", "den")

    def __init__(self, kind, sign, num=0, den=1):
        self.kind = kind          # 'num', 'inf', 'qnan', 'snan'
        self.sign = sign
        self.num = num
        self.den = den

    @property
    def is_nan(self):
        return self.kind in ("qnan", "snan")

    @property
    def is_zero(self):
        return self.kind == "num" and self.num == 0


def _mag(sig, e2):
    """sig*2**e2 as (num, den)."""
    if e2 >= 0:
        return sig << e2, 1
    return sig, 1 << -e2


def decode_f8(ui, fmt_code):
    f = FORMATS[fmt_code]
    ui &= 0xFF
    sign = bool(ui & 0x80)
    exp = (ui >> f.mant) & ((1 << (7 - f.mant)) - 1)
    frac = ui & ((1 << f.mant) - 1)
    if fmt_code == E4M3:
        if (ui & 0x7F) == 0x7F:
            return Val("qnan", sign)
    else:
        if exp == 0x1F:
            if not frac:
                return Val("inf", sign)
            return Val("snan" if frac == 1 else "qnan", sign)
    if exp == 0:
        return Val("num", sign, *_mag(frac, f.emin - f.mant))
    return Val("num", sign, *_mag(frac | (1 << f.mant), exp - f.bias - f.mant))


def decode_f32(bits):
    bits &= 0xFFFFFFFF
    sign = bool(bits >> 31)
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0xFF:
        if not frac:
            return Val("inf", sign)
        return Val("qnan" if frac & 0x400000 else "snan", sign)
    if exp == 0:
        return Val("num", sign, *_mag(frac, -149))
    return Val("num", sign, *_mag(frac | 0x800000, exp - 127 - 23))


# ---------------------------------------------------------------------------
# operations
# ---------------------------------------------------------------------------

def _inf(fmt_code, sign):
    return (0x80 if sign else 0) | FORMATS[fmt_code].inf_ui


def _nan(fmt_code):
    return FORMATS[fmt_code].nan_ui


def _nan_prologue(fmt_code, *vals):
    """sNaN raises invalid, any NaN gives the default NaN; None if no NaN."""
    if any(v.kind == "snan" for v in vals):
        return _nan(fmt_code), NV
    if any(v.kind == "qnan" for v in vals):
        return _nan(fmt_code), 0
    return None


def _addsub(a, b, fmt_code, mode, negate_b):
    sb = (not b.sign) if negate_b else b.sign
    pre = _nan_prologue(fmt_code, a, b)
    if pre:
        return pre
    if a.kind == "inf" or b.kind == "inf":
        if a.kind == "inf" and b.kind == "inf":
            if a.sign != sb:
                return _nan(fmt_code), NV
            return _inf(fmt_code, a.sign), 0
        return (_inf(fmt_code, a.sign if a.kind == "inf" else sb), 0)

    num = (-a.num if a.sign else a.num) * b.den \
        + (-b.num if sb else b.num) * a.den
    den = a.den * b.den
    if num == 0:
        sign = a.sign if a.sign == sb else (mode == RDN)
        return (0x80 if sign else 0), 0
    sign = num < 0
    return round_to_f8(sign, abs(num), den, fmt_code, mode)


def op_add(ua, ub, fmt_code, mode):
    return _addsub(decode_f8(ua, fmt_code), decode_f8(ub, fmt_code), fmt_code,
                   mode, False)


def op_sub(ua, ub, fmt_code, mode):
    return _addsub(decode_f8(ua, fmt_code), decode_f8(ub, fmt_code), fmt_code,
                   mode, True)


def op_mul(ua, ub, fmt_code, mode):
    a, b = decode_f8(ua, fmt_code), decode_f8(ub, fmt_code)
    pre = _nan_prologue(fmt_code, a, b)
    if pre:
        return pre
    sign = a.sign != b.sign
    if a.kind == "inf" or b.kind == "inf":
        if a.is_zero or b.is_zero:
            return _nan(fmt_code), NV
        return _inf(fmt_code, sign), 0
    return round_to_f8(sign, a.num * b.num, a.den * b.den, fmt_code, mode)


def op_div(ua, ub, fmt_code, mode):
    a, b = decode_f8(ua, fmt_code), decode_f8(ub, fmt_code)
    pre = _nan_prologue(fmt_code, a, b)
    if pre:
        return pre
    sign = a.sign != b.sign
    if a.kind == "inf":
        if b.kind == "inf":
            return _nan(fmt_code), NV
        return _inf(fmt_code, sign), 0
    if b.kind == "inf":
        return (0x80 if sign else 0), 0
    if b.is_zero:
        if a.is_zero:
            return _nan(fmt_code), NV
        return _inf(fmt_code, sign), DZ
    return round_to_f8(sign, a.num * b.den, a.den * b.num, fmt_code, mode)


def op_narrow(bits, fmt_code, mode):
    a = decode_f32(bits)
    pre = _nan_prologue(fmt_code, a)
    if pre:
        return pre
    if a.kind == "inf":
        return _inf(fmt_code, a.sign), 0
    return round_to_f8(a.sign, a.num, a.den, fmt_code, mode)


def op_widen(ua, fmt_code):
    """f8_to_f32 is exact; only a signaling NaN operand raises."""
    return NV if decode_f8(ua, fmt_code).kind == "snan" else 0


def cmp_flags(ua, ub, fmt_code, signaling):
    """eq is quiet (sNaN only); lt and le signal on any NaN."""
    a, b = decode_f8(ua, fmt_code), decode_f8(ub, fmt_code)
    if a.kind == "snan" or b.kind == "snan":
        return NV
    if signaling and (a.is_nan or b.is_nan):
        return NV
    return 0
