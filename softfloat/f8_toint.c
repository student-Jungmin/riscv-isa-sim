
#include <stdbool.h>
#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "specialize.h"
#include "softfloat.h"

/*----------------------------------------------------------------------------
| Routed through f16, which holds every fp8 value exactly, so the saturation
| and flag behaviour is the f16 routines' unchanged.  f32 lacks the 8- and
| 16-bit integer entry points these need.
*----------------------------------------------------------------------------*/
uint_fast8_t f8_to_ui8( float8_t a, uint_fast8_t roundingMode, bool exact )
{
    return f16_to_ui8( f8_to_f16( a ), roundingMode, exact );
}

uint_fast16_t f8_to_ui16( float8_t a, uint_fast8_t roundingMode, bool exact )
{
    return f16_to_ui16( f8_to_f16( a ), roundingMode, exact );
}

uint_fast32_t f8_to_ui32( float8_t a, uint_fast8_t roundingMode, bool exact )
{
    return f16_to_ui32( f8_to_f16( a ), roundingMode, exact );
}

uint_fast64_t f8_to_ui64( float8_t a, uint_fast8_t roundingMode, bool exact )
{
    return f16_to_ui64( f8_to_f16( a ), roundingMode, exact );
}

int_fast8_t f8_to_i8( float8_t a, uint_fast8_t roundingMode, bool exact )
{
    return f16_to_i8( f8_to_f16( a ), roundingMode, exact );
}

int_fast16_t f8_to_i16( float8_t a, uint_fast8_t roundingMode, bool exact )
{
    return f16_to_i16( f8_to_f16( a ), roundingMode, exact );
}

int_fast32_t f8_to_i32( float8_t a, uint_fast8_t roundingMode, bool exact )
{
    return f16_to_i32( f8_to_f16( a ), roundingMode, exact );
}

int_fast64_t f8_to_i64( float8_t a, uint_fast8_t roundingMode, bool exact )
{
    return f16_to_i64( f8_to_f16( a ), roundingMode, exact );
}

uint_fast32_t f8_to_ui32_r_minMag( float8_t a, bool exact )
{
    return f16_to_ui32_r_minMag( f8_to_f16( a ), exact );
}

uint_fast64_t f8_to_ui64_r_minMag( float8_t a, bool exact )
{
    return f16_to_ui64_r_minMag( f8_to_f16( a ), exact );
}

int_fast32_t f8_to_i32_r_minMag( float8_t a, bool exact )
{
    return f16_to_i32_r_minMag( f8_to_f16( a ), exact );
}

int_fast64_t f8_to_i64_r_minMag( float8_t a, bool exact )
{
    return f16_to_i64_r_minMag( f8_to_f16( a ), exact );
}
