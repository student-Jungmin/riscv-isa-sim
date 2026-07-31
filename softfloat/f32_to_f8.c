
#include <stdbool.h>
#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "specialize.h"
#include "softfloat.h"

/*----------------------------------------------------------------------------
| The one narrowing step, and so the only place fp8 rounding and fp8-relative
| inexact/overflow/underflow may be raised.  E4M3 has no infinity to overflow
| into, so it overflows onto S.1111.111, the encoding it also uses for NaN.
*----------------------------------------------------------------------------*/
float8_t f32_to_f8( float32_t a )
{
    union ui32_f32 uA;
    uint_fast32_t uiA;
    bool sign;
    int_fast16_t exp, expOffset, expMax;
    uint_fast32_t frac;
    struct commonNaN commonNaN;
    uint_fast8_t uiZ, uiZOverflow, mantBits, shiftDist;
    uint_fast16_t hiddenBit, carryBit, sigOverflow, sig;
    uint_fast8_t roundingMode, roundIncrement, roundBits;
    bool roundNearEven, isTiny;
    union ui8_f8 uZ;

    /*------------------------------------------------------------------------
    | Four round bits sit below the stored mantissa, so 'sig' holds the hidden
    | bit at 'hiddenBit' and rounding works off a 0xF mask either way.
    *------------------------------------------------------------------------*/
    if ( softfloat_fp8Format == softfloat_fp8_e4m3 ) {
        mantBits = 3;
        shiftDist = 16;
        hiddenBit = 0x80;
        expOffset = 0x79;
        expMax = 0x0E;
        sigOverflow = 0x0F;
    } else {
        mantBits = 2;
        shiftDist = 17;
        hiddenBit = 0x40;
        expOffset = 0x71;
        expMax = 0x1D;
        sigOverflow = 0x08;
    }
    carryBit = hiddenBit<<1;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uA.f = a;
    uiA = uA.ui;
    sign = signF32UI( uiA );
    exp  = expF32UI( uiA );
    frac = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    | Lacking an infinity, E4M3 sends both infinities and every overflow to
    | S.1111.111, matching torch.float8_e4m3fn.
    *------------------------------------------------------------------------*/
    uiZOverflow =
        ((uint_fast8_t) sign<<7)
            | (softfloat_fp8Format == softfloat_fp8_e4m3 ? 0x7F : 0x7C);
    if ( exp == 0xFF ) {
        if ( frac ) {
            softfloat_f32UIToCommonNaN( uiA, &commonNaN );
            uiZ = softfloat_commonNaNToF8UI( &commonNaN );
        } else {
            uiZ = uiZOverflow;
        }
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig =
        frac>>shiftDist
            | ((frac & (((uint_fast32_t) 1<<shiftDist) - 1)) != 0);
    if ( ! (exp | sig) ) {
        uiZ = (uint_fast8_t) sign<<7;
        goto uiZ;
    }
    sig |= hiddenBit;
    exp -= expOffset;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundingMode = softfloat_roundingMode;
    roundNearEven = (roundingMode == softfloat_round_near_even);
    roundIncrement = 0x8;
    if ( ! roundNearEven && (roundingMode != softfloat_round_near_maxMag) ) {
        roundIncrement =
            (roundingMode
                 == (sign ? softfloat_round_min : softfloat_round_max))
                ? 0xF
                : 0;
    }
    roundBits = sig & 0xF;
    /*------------------------------------------------------------------------
    | Tininess is judged against the fp8 minimum normal, 2^-6 or 2^-14, which
    | every f32 subnormal and a great many f32 normals fall below.
    *------------------------------------------------------------------------*/
    if ( exp < 0 ) {
        isTiny =
            (softfloat_detectTininess == softfloat_tininess_beforeRounding)
                || (exp < -1) || (sig + roundIncrement < carryBit);
        sig = softfloat_shiftRightJam32( sig, -exp );
        exp = 0;
        roundBits = sig & 0xF;
        if ( isTiny && roundBits ) {
            softfloat_raiseFlags( softfloat_flag_underflow );
        }
    }
    /*------------------------------------------------------------------------
    | Rounding precedes the overflow test because E4M3 shares its largest
    | exponent with its NaN:  the tie at 464 has to settle on 448 first, or a
    | value still inside the finite range would be reported as overflowing.
    *------------------------------------------------------------------------*/
    sig = (sig + roundIncrement)>>4;
    sig &= ~(uint_fast16_t) (! (roundBits ^ 8) & roundNearEven);
#ifdef SOFTFLOAT_ROUND_ODD
    /*------------------------------------------------------------------------
    | E4M3 has no odd significand to jam to at its largest exponent, 448 being
    | even and 480 the NaN, so the jam happens before the overflow test and
    | leaves 448 the way round-toward-zero would.
    *------------------------------------------------------------------------*/
    if ( roundBits && (roundingMode == softfloat_round_odd) ) sig |= 1;
#endif
    if ( (expMax < exp) || ((exp == expMax) && (sigOverflow <= sig)) ) {
        softfloat_raiseFlags(
            softfloat_flag_overflow | softfloat_flag_inexact );
        uiZ = uiZOverflow - ! roundIncrement;
        goto uiZ;
    }
    if ( roundBits ) softfloat_exceptionFlags |= softfloat_flag_inexact;
    if ( ! sig ) exp = 0;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiZ =
        ((uint_fast8_t) sign<<7) + ((uint_fast8_t) exp<<mantBits)
            + (uint_fast8_t) sig;
 uiZ:
    uZ.ui = uiZ;
    return uZ.f;

}
