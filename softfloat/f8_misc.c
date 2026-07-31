
#include <stdbool.h>
#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "specialize.h"
#include "softfloat.h"

/*----------------------------------------------------------------------------
| The three that cannot be derived from f32.  classify's 10-bit mask has no
| infinity bits to set under E4M3, and a 7-bit estimate is wider than either
| format's significand, so all three need fp8 semantics of their own.
*----------------------------------------------------------------------------*/
uint_fast16_t f8_classify( float8_t a )
{
    union ui8_f8 uA;
    uint_fast8_t uiA;

    uA.f = a;
    uiA = uA.ui;

    bool e4m3 = softfloat_fp8Format == softfloat_fp8_e4m3;
    uint_fast8_t exp = e4m3 ? (uiA>>3) & 0x0F : (uiA>>2) & 0x1F;
    uint_fast8_t frac = e4m3 ? uiA & 0x07 : uiA & 0x03;
    uint_fast16_t infOrNaN = e4m3 ? (uiA & 0x7F) == 0x7F : exp == 0x1F;
    uint_fast16_t subnormalOrZero = exp == 0;
    bool sign = (uiA>>7) & 1;
    bool fracZero = frac == 0;
    bool isNaN = infOrNaN && !fracZero;
    bool isSNaN = softfloat_isSigNaNF8UI( uiA );

    return
        (  sign && infOrNaN && fracZero )          << 0 |
        (  sign && !infOrNaN && !subnormalOrZero ) << 1 |
        (  sign && subnormalOrZero && !fracZero )  << 2 |
        (  sign && subnormalOrZero && fracZero )   << 3 |
        ( !sign && infOrNaN && fracZero )          << 7 |
        ( !sign && !infOrNaN && !subnormalOrZero ) << 6 |
        ( !sign && subnormalOrZero && !fracZero )  << 5 |
        ( !sign && subnormalOrZero && fracZero )   << 4 |
        ( isNaN &&  isSNaN )                       << 8 |
        ( isNaN && !isSNaN )                       << 9;
}

/*----------------------------------------------------------------------------
| An infinite estimate, in a format that may have no infinity:  E4M3 lands on
| S.1111.111, the same encoding f32_to_f8 gives infinity and overflow.
*----------------------------------------------------------------------------*/
static uint_fast8_t f8_infUI( bool sign )
{

    return
        ((uint_fast8_t) sign<<7)
            | (softfloat_fp8Format == softfloat_fp8_e4m3 ? 0x7F : 0x7C);
}

/*----------------------------------------------------------------------------
| The estimate itself:  widen to f16 (exact for every fp8 value of either
| format), run the RVV 7-bit table there, and round back down.  The narrowing's
| own inexact/underflow are dropped; RVV reports only its overflow, with it.
*----------------------------------------------------------------------------*/
static float8_t f8_narrowEstimate( float16_t a16 )
{
    union ui8_f8 uZ;
    uint_fast8_t savedFlags;

    savedFlags = softfloat_exceptionFlags;
    softfloat_exceptionFlags = 0;
    uZ.f = f16_to_f8( a16 );
    if ( softfloat_exceptionFlags & softfloat_flag_overflow ) {
        savedFlags |= softfloat_flag_overflow | softfloat_flag_inexact;
    }
    softfloat_exceptionFlags = savedFlags;
    return uZ.f;
}

float8_t f8_recip7( float8_t a )
{
    union ui8_f8 uZ;

    switch ( f8_classify( a ) ) {
     case 0x001: /* -inf */
        uZ.ui = 0x80;
        break;
     case 0x080: /* +inf */
        uZ.ui = 0x00;
        break;
     case 0x008: /* -0 */
        uZ.ui = f8_infUI( true );
        softfloat_exceptionFlags |= softfloat_flag_infinite;
        break;
     case 0x010: /* +0 */
        uZ.ui = f8_infUI( false );
        softfloat_exceptionFlags |= softfloat_flag_infinite;
        break;
     case 0x100: /* sNaN */
        softfloat_exceptionFlags |= softfloat_flag_invalid;
     case 0x200: /* qNaN */
        uZ.ui = defaultNaNF8UI;
        break;
     default: /* +-normal, +-subnormal */
        return f8_narrowEstimate( f16_recip7( f8_to_f16( a ) ) );
    }

    return uZ.f;
}

float8_t f8_rsqrte7( float8_t a )
{
    union ui8_f8 uZ;

    switch ( f8_classify( a ) ) {
     case 0x001: /* -inf */
     case 0x002: /* -normal */
     case 0x004: /* -subnormal */
     case 0x100: /* sNaN */
        softfloat_exceptionFlags |= softfloat_flag_invalid;
     case 0x200: /* qNaN */
        uZ.ui = defaultNaNF8UI;
        break;
     case 0x008: /* -0 */
        uZ.ui = f8_infUI( true );
        softfloat_exceptionFlags |= softfloat_flag_infinite;
        break;
     case 0x010: /* +0 */
        uZ.ui = f8_infUI( false );
        softfloat_exceptionFlags |= softfloat_flag_infinite;
        break;
     case 0x080: /* +inf */
        uZ.ui = 0x00;
        break;
     default: /* +normal, +subnormal */
        return f8_narrowEstimate( f16_rsqrte7( f8_to_f16( a ) ) );
    }

    return uZ.f;
}
