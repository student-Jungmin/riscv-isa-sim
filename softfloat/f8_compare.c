
#include <stdbool.h>
#include <stdint.h>
#include "platform.h"
#include "internals.h"
#include "specialize.h"
#include "softfloat.h"

/*----------------------------------------------------------------------------
| Widening to f32 is exact and order-preserving, so the f32 predicate answers
| for fp8 too -- including which NaN encodings signal, via f8_to_f32.
*----------------------------------------------------------------------------*/
bool f8_eq( float8_t a, float8_t b )
{
    return f32_eq( f8_to_f32( a ), f8_to_f32( b ) );
}

bool f8_le( float8_t a, float8_t b )
{
    return f32_le( f8_to_f32( a ), f8_to_f32( b ) );
}

bool f8_lt( float8_t a, float8_t b )
{
    return f32_lt( f8_to_f32( a ), f8_to_f32( b ) );
}

bool f8_eq_signaling( float8_t a, float8_t b )
{
    return f32_eq_signaling( f8_to_f32( a ), f8_to_f32( b ) );
}

bool f8_le_quiet( float8_t a, float8_t b )
{
    return f32_le_quiet( f8_to_f32( a ), f8_to_f32( b ) );
}

bool f8_lt_quiet( float8_t a, float8_t b )
{
    return f32_lt_quiet( f8_to_f32( a ), f8_to_f32( b ) );
}

/*----------------------------------------------------------------------------
| Answered on the fp8 bits directly: E4M3 has no signaling encoding at all.
*----------------------------------------------------------------------------*/
bool f8_isSignalingNaN( float8_t a )
{
    union ui8_f8 uA;

    uA.f = a;
    return softfloat_isSigNaNF8UI( uA.ui );
}
