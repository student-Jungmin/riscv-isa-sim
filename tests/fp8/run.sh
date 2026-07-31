#!/usr/bin/env bash
# Regenerate the fp8 goldens, build the driver, run it, print the summary.
# Usable with no arguments; knobs are FP8_PY, FP8_MAX_REPORT, FP8_SKIP_GEN.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
BUILD="$ROOT/build"
GOLDEN="$HERE/golden.bin"
DRIVER="$HERE/test_fp8"

die() { echo "run.sh: $*" >&2; exit 2; }

[ -f "$BUILD/libsoftfloat.so" ] || die "no $BUILD/libsoftfloat.so -- build spike first"

# RUNPATH loses to LD_LIBRARY_PATH, and /release/lib holds a pre-fp8 libsoftfloat
export LD_LIBRARY_PATH="$BUILD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

pick_python() {
    if [ -n "${FP8_PY:-}" ]; then echo "$FP8_PY"; return; fi
    for p in /usr/bin/python3.11 /workspace/torch210-env/bin/python3 python3; do
        if command -v "$p" >/dev/null 2>&1 &&
           "$p" -c 'import torch,numpy' >/dev/null 2>&1; then
            echo "$p"; return
        fi
    done
}

echo "=== 1/4 golden generation ==="
if [ "${FP8_SKIP_GEN:-0}" = "1" ] && [ -f "$GOLDEN" ]; then
    echo "skipped (FP8_SKIP_GEN=1), reusing $GOLDEN"
else
    PY=$(pick_python)
    [ -n "$PY" ] || die "no python with torch+numpy found; set FP8_PY"
    "$PY" "$HERE/gen_golden.py" -o "$GOLDEN" || die "golden generation failed"
fi

echo
echo "=== 2/4 build ==="
CC=${CC:-cc}
# DT_RPATH (old dtags) outranks LD_LIBRARY_PATH, so the export above is belt
# and braces rather than the only defence
"$CC" -O2 -std=c11 -Wall -Wextra \
    -I"$HERE" -I"$ROOT/softfloat" \
    "$HERE/test_fp8.c" -o "$DRIVER" \
    -L"$BUILD" -lsoftfloat \
    -Wl,-rpath,"$BUILD" -Wl,--disable-new-dtags -lm \
    || die "compile failed"
echo "built $DRIVER"

echo
echo "=== 3/4 library check ==="
RESOLVED=$(ldd "$DRIVER" | awk '/libsoftfloat\.so/ {print $3}')
[ -n "$RESOLVED" ] || die "ldd could not resolve libsoftfloat.so for $DRIVER"
echo "libsoftfloat.so -> $RESOLVED"
MISSING=""
for sym in f8_to_f32 f32_to_f8 f8_add f8_div f8_lt softfloat_fp8Format; do
    nm -D --defined-only "$RESOLVED" 2>/dev/null | grep -qw "$sym" || MISSING="$MISSING $sym"
done
if [ -n "$MISSING" ]; then
    echo "run.sh: the loaded libsoftfloat.so is missing fp8 symbols:$MISSING" >&2
    echo "run.sh: resolved to $RESOLVED -- a stale library shadowed $BUILD." >&2
    echo "run.sh: LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >&2
    exit 2
fi
echo "all fp8 symbols present"

echo
echo "=== 4/4 run ==="
"$DRIVER" "$GOLDEN"
rc=$?
echo
if [ $rc -eq 0 ]; then
    echo "run.sh: PASS"
else
    echo "run.sh: FAIL (exit $rc) -- mismatches above"
fi
exit $rc
