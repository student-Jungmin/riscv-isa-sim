
#include <stdbool.h>
#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "specialize.h"
#include "softfloat.h"

/*----------------------------------------------------------------------------
| Via f32.  Rounding an integer to 24 bits and then to fp8 lands on the same
| value as rounding it straight to fp8, again because 24 >= 2p+2.
*----------------------------------------------------------------------------*/
float8_t ui32_to_f8( uint32_t a )
{
    return f32_to_f8( ui32_to_f32( a ) );
}

float8_t ui64_to_f8( uint64_t a )
{
    return f32_to_f8( ui64_to_f32( a ) );
}

float8_t i32_to_f8( int32_t a )
{
    return f32_to_f8( i32_to_f32( a ) );
}

float8_t i64_to_f8( int64_t a )
{
    return f32_to_f8( i64_to_f32( a ) );
}
