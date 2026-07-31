
#include <stdbool.h>
#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "specialize.h"
#include "softfloat.h"

/*----------------------------------------------------------------------------
| Widening never rounds -- every fp8 value is exact in f16, f32 and f64 alike.
*----------------------------------------------------------------------------*/
float16_t f8_to_f16( float8_t a )
{
    return f32_to_f16( f8_to_f32( a ) );
}

float64_t f8_to_f64( float8_t a )
{
    return f32_to_f64( f8_to_f32( a ) );
}

/*----------------------------------------------------------------------------
| Narrowing.  f16 is exact in f32, so f16_to_f8 rounds once, in f32_to_f8.
| f64_to_f8 rounds at f32 first, which 24 >= 2p+2 makes innocuous.
*----------------------------------------------------------------------------*/
float8_t f16_to_f8( float16_t a )
{
    return f32_to_f8( f16_to_f32( a ) );
}

float8_t f64_to_f8( float64_t a )
{
    return f32_to_f8( f64_to_f32( a ) );
}
