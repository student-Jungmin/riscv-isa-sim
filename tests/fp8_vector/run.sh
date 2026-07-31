#!/usr/bin/env bash
# Differential test of the SEW=8 fp8 vector instructions on spike.
# Builds the target, runs it under pk, diffs the stream against two
# independent references, then runs the trap and negative-control checks.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BUILD="$ROOT/build"
OUT=${FP8V_OUT:-"$HERE/run.bin"}

SPIKE=${SPIKE:-"$BUILD/spike"}
PK=${PK:-/workspace/riscv-pk/build/pk}
CROSS=${CROSS:-/workspace/riscv/bin/riscv64-unknown-elf-gcc}
ISA=${ISA:-rv64gcv_zvfp8_zfh}

die() { echo "run.sh: $*" >&2; exit 2; }
[ -x "$SPIKE" ] || die "no spike at $SPIKE"
[ -x "$PK" ] || die "no pk at $PK"
[ -x "$CROSS" ] || die "no cross compiler at $CROSS"
[ -f "$BUILD/libsoftfloat.so" ] || die "no $BUILD/libsoftfloat.so"

# RUNPATH loses to LD_LIBRARY_PATH and /release/lib holds a pre-fp8 build
export LD_LIBRARY_PATH="$BUILD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CFLAGS_T="-march=rv64gcv_zfh -mabi=lp64d -O2 -std=gnu11 -Wall -Wextra -I$HERE"
rc_total=0

echo "=== 1/6 build ==="
$CROSS $CFLAGS_T "$HERE/target.c" -o "$HERE/target" || die "target build failed"
$CROSS $CFLAGS_T -DFP8V_SABOTAGE=1 "$HERE/target.c" -o "$HERE/target_sab" \
    || die "sabotage target build failed"
${CC:-cc} -O2 -std=gnu11 -Wall -Wextra -I"$HERE" -I"$ROOT/softfloat" \
    "$HERE/ref.c" -o "$HERE/ref" -L"$BUILD" -lsoftfloat \
    -Wl,-rpath,"$BUILD" -Wl,--disable-new-dtags -lm || die "ref build failed"

RESOLVED=$(ldd "$HERE/ref" | awk '/libsoftfloat\.so/ {print $3}')
for sym in f8_add f8_mulAdd f8_recip7 f16_to_f8; do
    nm -D --defined-only "$RESOLVED" 2>/dev/null | grep -qw "$sym" \
        || die "loaded libsoftfloat.so ($RESOLVED) lacks $sym -- stale library"
done
echo "libsoftfloat.so -> $RESOLVED"

echo
echo "=== 2/6 run on spike ==="
"$SPIKE" --isa="$ISA" "$PK" "$HERE/target" > "$OUT"
srv=$?
[ $srv -eq 0 ] || die "target exited $srv -- an instruction trapped"
tail -c 12 "$OUT" | grep -q ENDFP8VEC || die "stream truncated"
echo "wrote $OUT ($(stat -c %s "$OUT") bytes), stream terminated cleanly"

echo
echo "=== 3/6 softfloat reference ==="
"$HERE/ref" "$OUT"
[ $? -eq 0 ] || rc_total=1

echo
echo "=== 4/6 exact-rational oracle (independent of softfloat) ==="
PY=${FP8V_PY:-python3}
"$PY" "$HERE/oracle_check.py" "$OUT" ${FP8V_ORACLE_ARGS:-}
[ $? -eq 0 ] || rc_total=1

echo
echo "=== 5/6 negative controls (each MUST fail) ==="
nc_ok=1
FP8V_MUTATE=1000 FP8V_MAX_REPORT=0 "$HERE/ref" "$OUT" >/dev/null 2>&1
[ $? -ne 0 ] && echo "  ok   ref catches a mutated expectation" \
             || { echo "  BAD  ref passed a mutated expectation"; nc_ok=0; }
"$PY" "$HERE/oracle_check.py" "$OUT" --mutate 997 --max-report 0 \
    >/dev/null 2>&1
[ $? -ne 0 ] && echo "  ok   oracle catches a mutated expectation" \
             || { echo "  BAD  oracle passed a mutated expectation"; nc_ok=0; }
"$SPIKE" --isa="$ISA" "$PK" "$HERE/target_sab" > "$HERE/sabotage.bin" 2>/dev/null
sab=$("$HERE/ref" "$HERE/sabotage.bin" 2>/dev/null)
if echo "$sab" | grep -q "^vfsub.vv .*FAIL"; then
    echo "  ok   ref catches vfsub.vv built with swapped operands"
else
    echo "  BAD  swapped-operand vfsub.vv went unnoticed"; nc_ok=0
fi
rm -f "$HERE/sabotage.bin"
[ $nc_ok -eq 1 ] || rc_total=1

echo
echo "=== 6/6 trap and gating checks ==="
# name : -DT : isa : expect (trap|run)
CASES="
vfclass.v@e8:0:rv64gcv_zvfp8_zfh:trap
vfadd.vv@e8_no_zvfp8:1:rv64gcv_zfh:trap
vfwadd.vv@e8_no_zfh:2:rv64gcv_zvfp8:trap
vfncvt.f.f.w@e8_no_zfh:3:rv64gcv_zvfp8:trap
vfwcvt.x.f.v@e8_no_zfh:8:rv64gcv_zvfp8:run
vfwredosum.vs@e8_no_zfh:9:rv64gcv_zvfp8:trap
vfwcvt.f.f.v@e8_no_zfh:4:rv64gcv_zvfp8:trap
vfredosum.vs@e8_no_zvfp8:5:rv64gcv_zfh:trap
vfncvt.x.f.w@e8_int:6:rv64gcv_zvfp8_zfh:run
vfwcvt.f.x.v@e8_int:7:rv64gcv_zvfp8_zfh:run
vfadd.vv@e8_control:1:rv64gcv_zvfp8_zfh:run
"
trap_ok=1
for c in $CASES; do
    nm=${c%%:*}; rest=${c#*:}
    t=${rest%%:*}; rest=${rest#*:}
    isa=${rest%%:*}; want=${rest##*:}
    $CROSS $CFLAGS_T -DT=$t "$HERE/traps.c" -o "$HERE/trap_bin" 2>/dev/null \
        || die "traps.c T=$t build failed"
    got=$("$SPIKE" --isa="$isa" "$PK" "$HERE/trap_bin" 2>&1)
    if echo "$got" | grep -q "AFTER"; then have=run; else have=trap; fi
    if [ "$have" = "$want" ]; then
        printf "  ok   %-32s isa=%-20s %s\n" "$nm" "$isa" "$have"
    else
        printf "  BAD  %-32s isa=%-20s want=%s got=%s\n" "$nm" "$isa" \
               "$want" "$have"
        trap_ok=0
    fi
done
rm -f "$HERE/trap_bin"
[ $trap_ok -eq 1 ] || rc_total=1

echo
if [ $rc_total -eq 0 ]; then echo "run.sh: PASS"; else echo "run.sh: FAIL"; fi
exit $rc_total
