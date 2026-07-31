
#include <stdbool.h>
#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "specialize.h"
#include "softfloat.h"

/*----------------------------------------------------------------------------
| Widening to f32 is always exact: every finite E4M3 and E5M2 value, subnormals
| included, is a normal f32.  Only the NaN/infinity mapping differs by format.
*----------------------------------------------------------------------------*/
float32_t f8_to_f32( float8_t a )
{
    union ui8_f8 uA;
    uint_fast8_t uiA;
    bool sign, isNaN, isInf;
    int_fast8_t exp;
    uint_fast8_t frac, hiddenBit, shiftDist;
    int_fast16_t expOffset;
    struct commonNaN commonNaN;
    uint_fast32_t uiZ;
    union ui32_f32 uZ;

    /*------------------------------------------------------------------------
    | E4M3 spends S.1111.111 on its only NaN and has no infinity; E5M2 keeps
    | the IEEE all-ones exponent for both NaN and infinity.
    *------------------------------------------------------------------------*/
    uA.f = a;
    uiA = uA.ui;
    sign = ((uiA & 0x80) != 0);
    if ( softfloat_fp8Format == softfloat_fp8_e4m3 ) {
        exp = (int_fast8_t) ((uiA>>3) & 0x0F);
        frac = uiA & 0x07;
        isNaN = ((uiA & 0x7F) == 0x7F);
        isInf = false;
        hiddenBit = 0x08;
        expOffset = 0x78;
        shiftDist = 20;
    } else {
        exp = (int_fast8_t) ((uiA>>2) & 0x1F);
        frac = uiA & 0x03;
        isNaN = (exp == 0x1F) && frac;
        isInf = (exp == 0x1F) && ! frac;
        hiddenBit = 0x04;
        expOffset = 0x70;
        shiftDist = 21;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( isNaN ) {
        softfloat_f8UIToCommonNaN( uiA, &commonNaN );
        uiZ = softfloat_commonNaNToF32UI( &commonNaN );
        goto uiZ;
    }
    if ( isInf ) {
        uiZ = packToF32UI( sign, 0xFF, 0 );
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    | A subnormal is an exponent-1 significand shifted right, so shift it back
    | up until the hidden bit appears, spending one exponent per step.
    *------------------------------------------------------------------------*/
    if ( ! exp ) {
        if ( ! frac ) {
            uiZ = packToF32UI( sign, 0, 0 );
            goto uiZ;
        }
        exp = 1;
        do {
            --exp;
            frac <<= 1;
        } while ( ! (frac & hiddenBit) );
        frac &= hiddenBit - 1;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiZ =
        packToF32UI( sign, exp + expOffset, (uint_fast32_t) frac<<shiftDist );
 uiZ:
    uZ.ui = uiZ;
    return uZ.f;

}
