
#include <stdbool.h>
#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "specialize.h"
#include "softfloat.h"

/*----------------------------------------------------------------------------
| Every operation evaluates in f32 and rounds once on the way back.  f32's 24
| bits clear the 2p+2 (10 for E4M3, 8 for E5M2) that makes the second rounding
| innocuous, so each result is the correctly rounded fp8 one.
*----------------------------------------------------------------------------*/
float8_t f8_add( float8_t a, float8_t b )
{
    return f32_to_f8( f32_add( f8_to_f32( a ), f8_to_f32( b ) ) );
}

float8_t f8_sub( float8_t a, float8_t b )
{
    return f32_to_f8( f32_sub( f8_to_f32( a ), f8_to_f32( b ) ) );
}

float8_t f8_mul( float8_t a, float8_t b )
{
    return f32_to_f8( f32_mul( f8_to_f32( a ), f8_to_f32( b ) ) );
}

float8_t f8_div( float8_t a, float8_t b )
{
    return f32_to_f8( f32_div( f8_to_f32( a ), f8_to_f32( b ) ) );
}

float8_t f8_rem( float8_t a, float8_t b )
{
    return f32_to_f8( f32_rem( f8_to_f32( a ), f8_to_f32( b ) ) );
}

float8_t f8_sqrt( float8_t a )
{
    return f32_to_f8( f32_sqrt( f8_to_f32( a ) ) );
}

/*----------------------------------------------------------------------------
| Fused, so the product is not rounded before the add.  Composing f32_mul with
| f32_add instead would round twice at f32 and step outside the 2p+2 argument.
*----------------------------------------------------------------------------*/
float8_t f8_mulAdd( float8_t a, float8_t b, float8_t c )
{
    return
        f32_to_f8(
            f32_mulAdd( f8_to_f32( a ), f8_to_f32( b ), f8_to_f32( c ) ) );
}

/*----------------------------------------------------------------------------
| These return one of their operands, so the narrowing back is exact.
*----------------------------------------------------------------------------*/
float8_t f8_min( float8_t a, float8_t b )
{
    return f32_to_f8( f32_min( f8_to_f32( a ), f8_to_f32( b ) ) );
}

float8_t f8_max( float8_t a, float8_t b )
{
    return f32_to_f8( f32_max( f8_to_f32( a ), f8_to_f32( b ) ) );
}

/*----------------------------------------------------------------------------
| Exact: an fp8 value at or above the format's unit spacing is already an
| integer, and one below it rounds into the small integers fp8 holds exactly.
*----------------------------------------------------------------------------*/
float8_t f8_roundToInt( float8_t a, uint_fast8_t roundingMode, bool exact )
{
    return f32_to_f8( f32_roundToInt( f8_to_f32( a ), roundingMode, exact ) );
}
