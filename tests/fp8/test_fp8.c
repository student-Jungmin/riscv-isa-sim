/*----------------------------------------------------------------------------
| Exhaustive fp8 conformance driver.  Replays every case in the golden file
| through the f8_* entry points and diffs raw bits.  Exits non-zero on any
| mismatch; see gen_golden.py for how the goldens are produced.
*----------------------------------------------------------------------------*/

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "softfloat.h"
#include "fp8_golden.h"

#define DEFAULT_MAX_REPORT 10
#define DEFAULT_MAX_NANFORM 3

struct section {
    char name[FP8_GOLDEN_NAME_LEN + 1];
    uint32_t op;
    uint32_t fmt;
    uint32_t count;
    const uint8_t *recs;
};

struct stats {
    unsigned long compared;
    unsigned long matched;
    unsigned long mismatched;
    unsigned long nanform;
    unsigned long flags_compared;
    unsigned long flags_mismatched;
};

/*----------------------------------------------------------------------------
| Little-endian field readers.  The wire format is fixed-width, so decoding by
| hand keeps the driver free of struct-packing and host-endianness assumptions.
*----------------------------------------------------------------------------*/
static uint32_t rd32( const uint8_t *p )
{
    return (uint32_t) p[0] | ( (uint32_t) p[1] << 8 )
        | ( (uint32_t) p[2] << 16 ) | ( (uint32_t) p[3] << 24 );
}

static const char *fmt_name( uint32_t fmt )
{
    return fmt == softfloat_fp8_e4m3 ? "e4m3"
        : fmt == softfloat_fp8_e5m2 ? "e5m2" : "?";
}

static const char *op_name( uint32_t op )
{
    switch ( op ) {
     case FP8_OP_WIDEN_F32: return "f8_to_f32";
     case FP8_OP_ADD:       return "f8_add";
     case FP8_OP_SUB:       return "f8_sub";
     case FP8_OP_MUL:       return "f8_mul";
     case FP8_OP_DIV:       return "f8_div";
     case FP8_OP_EQ:        return "f8_eq";
     case FP8_OP_LE:        return "f8_le";
     case FP8_OP_LT:        return "f8_lt";
     case FP8_OP_NARROW_F8: return "f32_to_f8";
    }
    return "?";
}

/*----------------------------------------------------------------------------
| NaN detection on raw bits, so a disagreement that is only about which NaN
| encoding comes out can be triaged apart from a wrong numeric result.
*----------------------------------------------------------------------------*/
static bool f8_bits_nan( uint32_t bits, uint32_t fmt )
{
    uint8_t u = (uint8_t) bits;

    if ( fmt == softfloat_fp8_e4m3 ) return ( u & 0x7F ) == 0x7F;
    return ( u & 0x7C ) == 0x7C && ( u & 0x03 ) != 0;
}

static bool f32_bits_nan( uint32_t bits )
{
    return ( bits & 0x7F800000 ) == 0x7F800000
        && ( bits & 0x007FFFFF ) != 0;
}

static bool out_is_f32( uint32_t op ) { return op == FP8_OP_WIDEN_F32; }

static bool out_is_f8( uint32_t op )
{
    return op == FP8_OP_ADD || op == FP8_OP_SUB || op == FP8_OP_MUL
        || op == FP8_OP_DIV || op == FP8_OP_NARROW_F8;
}

static float8_t mk_f8( uint32_t bits )
{
    float8_t z;

    z.v = (uint8_t) bits;
    return z;
}

static float32_t mk_f32( uint32_t bits )
{
    float32_t z;

    z.v = bits;
    return z;
}

/*----------------------------------------------------------------------------
| One case.  Flags are cleared before and captured after so the Phase 4 check
| is a comparison away, not a restructure.
*----------------------------------------------------------------------------*/
static uint32_t run_case(
    uint32_t op, uint32_t fmt, uint32_t i0, uint32_t i1, uint8_t rmode,
    uint8_t *flags_out )
{
    uint32_t actual = 0;

    softfloat_fp8Format = (uint_fast8_t) fmt;
    softfloat_roundingMode = (uint_fast8_t) rmode;
    softfloat_detectTininess = softfloat_tininess_afterRounding;
    softfloat_exceptionFlags = 0;

    switch ( op ) {
     case FP8_OP_WIDEN_F32:
        actual = f8_to_f32( mk_f8( i0 ) ).v;
        break;
     case FP8_OP_ADD:
        actual = f8_add( mk_f8( i0 ), mk_f8( i1 ) ).v;
        break;
     case FP8_OP_SUB:
        actual = f8_sub( mk_f8( i0 ), mk_f8( i1 ) ).v;
        break;
     case FP8_OP_MUL:
        actual = f8_mul( mk_f8( i0 ), mk_f8( i1 ) ).v;
        break;
     case FP8_OP_DIV:
        actual = f8_div( mk_f8( i0 ), mk_f8( i1 ) ).v;
        break;
     case FP8_OP_EQ:
        actual = f8_eq( mk_f8( i0 ), mk_f8( i1 ) ) ? 1 : 0;
        break;
     case FP8_OP_LE:
        actual = f8_le( mk_f8( i0 ), mk_f8( i1 ) ) ? 1 : 0;
        break;
     case FP8_OP_LT:
        actual = f8_lt( mk_f8( i0 ), mk_f8( i1 ) ) ? 1 : 0;
        break;
     case FP8_OP_NARROW_F8:
        actual = f32_to_f8( mk_f32( i0 ) ).v;
        break;
    }

    *flags_out = (uint8_t) softfloat_exceptionFlags;
    return actual;
}

static void print_inputs( char *buf, size_t n, uint32_t op, uint32_t i0,
                          uint32_t i1 )
{
    if ( op == FP8_OP_WIDEN_F32 ) {
        snprintf( buf, n, "a=0x%02x", i0 & 0xFF );
    } else if ( op == FP8_OP_NARROW_F8 ) {
        snprintf( buf, n, "a=0x%08x", i0 );
    } else {
        snprintf( buf, n, "a=0x%02x b=0x%02x", i0 & 0xFF, i1 & 0xFF );
    }
}

static void print_value( char *buf, size_t n, uint32_t op, uint32_t bits )
{
    if ( out_is_f32( op ) ) snprintf( buf, n, "0x%08x", bits );
    else if ( out_is_f8( op ) ) snprintf( buf, n, "0x%02x", bits & 0xFF );
    else snprintf( buf, n, "%u", bits );
}

/*----------------------------------------------------------------------------
| Both sides say NaN but not with the same encoding.  RISC-V canonicalizes NaN
| where torch propagates sign and payload, so this is a convention difference,
| not a numeric error; it is counted apart and gated by FP8_STRICT_NAN.
*----------------------------------------------------------------------------*/
static bool nan_form_only( uint32_t op, uint32_t fmt, uint32_t expect,
                           uint32_t actual )
{
    if ( out_is_f32( op ) ) return f32_bits_nan( expect ) && f32_bits_nan( actual );
    if ( out_is_f8( op ) ) {
        return f8_bits_nan( expect, fmt ) && f8_bits_nan( actual, fmt );
    }
    return false;
}

static void report_case( const struct section *sec, const char *tag,
                         uint32_t in0, uint32_t in1, uint32_t expect,
                         uint32_t actual, uint8_t act_flags,
                         uint8_t exp_flags, uint8_t flags_known )
{
    char ins[64], exps[32], acts[32], fexp[16];

    print_inputs( ins, sizeof ins, sec->op, in0, in1 );
    print_value( exps, sizeof exps, sec->op, expect );
    print_value( acts, sizeof acts, sec->op, actual );
    if ( flags_known ) snprintf( fexp, sizeof fexp, "0x%02x", exp_flags );
    else snprintf( fexp, sizeof fexp, "n/a" );

    printf( "%-8s fmt=%s op=%-9s %-22s expect=%-10s actual=%-10s"
            " flags_actual=0x%02x flags_expect=%s\n",
            tag, fmt_name( sec->fmt ), op_name( sec->op ), ins, exps, acts,
            act_flags, fexp );
}

static int run_section( const struct section *sec, unsigned long max_report,
                        unsigned long max_nanform, bool strict_nan,
                        struct stats *st )
{
    unsigned long reported = 0, nan_reported = 0;
    uint32_t i;

    memset( st, 0, sizeof *st );

    for ( i = 0; i < sec->count; ++i ) {
        const uint8_t *r = sec->recs + (size_t) i * FP8_GOLDEN_REC_LEN;
        uint32_t in0 = rd32( r );
        uint32_t in1 = rd32( r + 4 );
        uint32_t expect = rd32( r + 12 );
        uint8_t exp_flags = r[16];
        uint8_t flags_known = r[17];
        uint8_t rmode = r[18];
        uint8_t act_flags = 0;
        uint32_t actual;
        bool bits_ok, flags_ok = true, nanform;

        actual = run_case( sec->op, sec->fmt, in0, in1, rmode, &act_flags );
        bits_ok = ( actual == expect );

        if ( flags_known ) {
            ++st->flags_compared;
            flags_ok = ( act_flags == exp_flags );
            if ( !flags_ok ) ++st->flags_mismatched;
        }

        ++st->compared;
        if ( bits_ok && flags_ok ) {
            ++st->matched;
            continue;
        }

        nanform = !bits_ok && flags_ok
            && nan_form_only( sec->op, sec->fmt, expect, actual );

        if ( nanform && !strict_nan ) {
            ++st->nanform;
            if ( nan_reported < max_nanform ) {
                report_case( sec, "NANFORM", in0, in1, expect, actual,
                             act_flags, exp_flags, flags_known );
                ++nan_reported;
                if ( nan_reported == max_nanform ) {
                    printf( "  ... further NaN-encoding differences in %s"
                            " suppressed (raise FP8_MAX_NANFORM)\n",
                            sec->name );
                }
            }
            continue;
        }

        ++st->mismatched;
        if ( nanform ) ++st->nanform;

        if ( reported < max_report ) {
            report_case( sec, bits_ok ? "FLAGBAD" : "MISMATCH", in0, in1,
                         expect, actual, act_flags, exp_flags, flags_known );
            ++reported;
            if ( reported == max_report ) {
                printf( "  ... further mismatches in %s suppressed"
                        " (raise FP8_MAX_REPORT)\n", sec->name );
            }
        }
    }

    return st->mismatched != 0;
}

static uint8_t *slurp( const char *path, size_t *len )
{
    FILE *fh = fopen( path, "rb" );
    uint8_t *buf;
    long n;

    if ( !fh ) return NULL;
    if ( fseek( fh, 0, SEEK_END ) != 0 ) { fclose( fh ); return NULL; }
    n = ftell( fh );
    if ( n < 0 ) { fclose( fh ); return NULL; }
    rewind( fh );
    buf = malloc( (size_t) n );
    if ( !buf ) { fclose( fh ); return NULL; }
    if ( fread( buf, 1, (size_t) n, fh ) != (size_t) n ) {
        free( buf );
        fclose( fh );
        return NULL;
    }
    fclose( fh );
    *len = (size_t) n;
    return buf;
}

/*----------------------------------------------------------------------------
| Guards against the stale libsoftfloat.so on LD_LIBRARY_PATH: if the loader
| picked a pre-fp8 build, f8_to_f32 of every encoding is identically zero.
*----------------------------------------------------------------------------*/
static bool linked_library_has_fp8( void )
{
    uint32_t i, nonzero = 0;

    softfloat_fp8Format = softfloat_fp8_e4m3;
    for ( i = 0; i < 256; ++i ) {
        if ( f8_to_f32( mk_f8( i ) ).v != 0 ) ++nonzero;
    }
    return nonzero != 0;
}

int main( int argc, char **argv )
{
    const char *path = argc > 1 ? argv[1] : getenv( "FP8_GOLDEN" );
    const char *env_max = getenv( "FP8_MAX_REPORT" );
    const char *env_nan = getenv( "FP8_MAX_NANFORM" );
    const char *env_strict = getenv( "FP8_STRICT_NAN" );
    unsigned long max_report = env_max ? strtoul( env_max, NULL, 0 )
                                       : DEFAULT_MAX_REPORT;
    unsigned long max_nanform = env_nan ? strtoul( env_nan, NULL, 0 )
                                        : DEFAULT_MAX_NANFORM;
    bool strict_nan = env_strict && strcmp( env_strict, "0" ) != 0;
    struct stats total;
    size_t len = 0, off;
    uint8_t *buf;
    uint32_t nsections, s;
    int failed = 0;

    if ( !path ) path = "golden.bin";

    buf = slurp( path, &len );
    if ( !buf ) {
        fprintf( stderr, "test_fp8: cannot read golden file '%s'\n", path );
        return 2;
    }
    if ( len < FP8_GOLDEN_MAGIC_LEN + 4
             || memcmp( buf, FP8_GOLDEN_MAGIC, FP8_GOLDEN_MAGIC_LEN ) != 0 ) {
        fprintf( stderr, "test_fp8: '%s' is not an fp8 golden file\n", path );
        free( buf );
        return 2;
    }

    if ( !linked_library_has_fp8() ) {
        fprintf( stderr,
                 "test_fp8: the loaded libsoftfloat has no working fp8 support"
                 " (f8_to_f32 is zero for all 256 encodings).\n"
                 "          A stale libsoftfloat.so probably shadowed the build"
                 " tree; put %s first on LD_LIBRARY_PATH.\n",
                 "/workspace/spike-fp8/build" );
        /* not fatal: with stubs in place this is also the legitimate state */
    }

    nsections = rd32( buf + FP8_GOLDEN_MAGIC_LEN );
    off = FP8_GOLDEN_MAGIC_LEN + 4;

    memset( &total, 0, sizeof total );
    printf( "golden: %s (%u sections)\n", path, nsections );
    printf( "NaN-encoding differences are %s (FP8_STRICT_NAN)\n\n",
            strict_nan ? "FATAL" : "counted separately, not fatal" );

    for ( s = 0; s < nsections; ++s ) {
        struct section sec;
        struct stats st;
        uint32_t reclen;
        size_t need;

        if ( off + FP8_GOLDEN_SECHDR_LEN > len ) {
            fprintf( stderr, "test_fp8: truncated section header\n" );
            free( buf );
            return 2;
        }
        memcpy( sec.name, buf + off, FP8_GOLDEN_NAME_LEN );
        sec.name[FP8_GOLDEN_NAME_LEN] = '\0';
        sec.op = rd32( buf + off + FP8_GOLDEN_NAME_LEN );
        sec.fmt = rd32( buf + off + FP8_GOLDEN_NAME_LEN + 4 );
        sec.count = rd32( buf + off + FP8_GOLDEN_NAME_LEN + 8 );
        reclen = rd32( buf + off + FP8_GOLDEN_NAME_LEN + 12 );
        off += FP8_GOLDEN_SECHDR_LEN;

        if ( reclen != FP8_GOLDEN_REC_LEN ) {
            fprintf( stderr, "test_fp8: section %s has record length %u,"
                             " expected %u\n",
                     sec.name, reclen, FP8_GOLDEN_REC_LEN );
            free( buf );
            return 2;
        }
        need = (size_t) sec.count * FP8_GOLDEN_REC_LEN;
        if ( off + need > len ) {
            fprintf( stderr, "test_fp8: truncated records in %s\n", sec.name );
            free( buf );
            return 2;
        }
        sec.recs = buf + off;
        off += need;

        failed |= run_section( &sec, max_report, max_nanform, strict_nan, &st );

        total.compared += st.compared;
        total.matched += st.matched;
        total.mismatched += st.mismatched;
        total.nanform += st.nanform;
        total.flags_compared += st.flags_compared;
        total.flags_mismatched += st.flags_mismatched;

        printf( "SUMMARY %-22s compared=%-7lu matched=%-7lu"
                " nan-form=%-7lu mismatched=%-7lu flags_compared=%-7lu"
                " flags_bad=%lu\n",
                sec.name, st.compared, st.matched, st.nanform, st.mismatched,
                st.flags_compared, st.flags_mismatched );
    }

    printf( "\nTOTAL compared=%lu matched=%lu nan-form=%lu mismatched=%lu\n",
            total.compared, total.matched, total.nanform, total.mismatched );
    printf( "FLAGS compared=%lu mismatched=%lu", total.flags_compared,
            total.flags_mismatched );
    if ( !total.flags_compared ) {
        printf( "  (no flag goldens: torch reports no exception state;"
                " see flags_for() in gen_golden.py)" );
    }
    printf( "\n%s\n", total.mismatched ? "RESULT: FAIL" : "RESULT: PASS" );

    free( buf );
    return failed ? 1 : 0;
}
