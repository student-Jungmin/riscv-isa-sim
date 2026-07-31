#!/usr/bin/env bash
# Build and run the fp8 DMA / format-selector test on the fp8 spike fork.
# Knobs: LANES, SPIKE, PK, RISCV_PREFIX, SPIKE_DEBUG.
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

SPIKE=${SPIKE:-$ROOT/build/spike}
PK=${PK:-/workspace/riscv-pk/build/pk}
PREFIX=${RISCV_PREFIX:-/workspace/riscv/bin/riscv64-unknown-elf-}
LANES=${LANES:-4}

SPAD_PADDR=0x70000000
SPAD_VADDR=0xD0000000
SPAD_SIZE=65536
DRAM_BASE=0x80000000
DRAM_SIZE=0x80000000

die() { echo "run.sh: $*" >&2; exit 2; }

[ -x "$SPIKE" ] || die "no spike at $SPIKE"
[ -f "$PK" ]    || die "no pk at $PK"

# Optional: print the host-softfloat values for the same operand pairs, so the
# hand-derived tables compiled into fp8_dma.c can be eyeballed for drift.
if [ "${ORACLE:-1}" = "1" ] && [ -f "$ROOT/build/libsoftfloat.so" ]; then
    if cc -O2 -I"$ROOT/softfloat" "$HERE/oracle.c" -o "$HERE/oracle" \
         -L"$ROOT/build" -lsoftfloat -Wl,-rpath,"$ROOT/build" \
         -Wl,--disable-new-dtags -lm 2>/dev/null; then
        echo "host softfloat oracle (compare with the tables in fp8_dma.c):"
        LD_LIBRARY_PATH="$ROOT/build" "$HERE/oracle" | sed 's/^/  /'
        echo
    fi
fi

ELF=$HERE/fp8_dma.elf
"${PREFIX}gcc" -O2 -std=c11 -Wall -Wextra -march=rv64gcv_zfh -mabi=lp64d \
    "$HERE/fp8_dma.c" "$HERE/kernel.S" -o "$ELF" || die "compile failed"

# Spike engages the vector lanes only while the PC is inside --kernel-addr.
read -r KLO KHI < <("${PREFIX}nm" "$ELF" | awk '
    $3=="k_begin"{lo=$1} $3=="k_end"{hi=$1} END{print lo, hi}')
[ -n "$KLO" ] && [ -n "$KHI" ] || die "could not find k_begin/k_end in $ELF"
echo "kernel PC range: 0x$KLO..0x$KHI"

# get_spad_size() is scratchpad_size * lanes, so the -m region must be that big.
BANKED=$(( SPAD_SIZE * LANES ))

CMD=("$SPIKE"
     "--isa=rv64gcv_zfh_zvfp8"
     "--varch=vlen:256,elen:64"
     "--vectorlane-size=$LANES"
     "-m${DRAM_BASE}:${DRAM_SIZE},${SPAD_PADDR}:$(printf '0x%x' "$BANKED")"
     # atoul_safe() is decimal-only: a 0x... value makes spike print usage.
     "--scratchpad-base-paddr=$((SPAD_PADDR))"
     "--scratchpad-base-vaddr=$((SPAD_VADDR))"
     "--scratchpad-size=$((SPAD_SIZE))"
     "--kernel-addr=${KLO}:${KHI}"
     "--base-path=$HERE"
     "$PK" "$ELF" "$LANES")

mkdir -p "$HERE/dma_access" "$HERE/indirect_access"
printf 'spike command:\n  %s\n\n' "${CMD[*]}"
"${CMD[@]}"
rc=$?
echo
echo "run.sh: exit $rc"
exit $rc
