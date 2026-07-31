#!/bin/bash
# Build and run the fp8 systolic-array end-to-end test under pk on this fork.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SPIKE=${SPIKE:-/workspace/spike-fp8/build/spike}
PK=${PK:-/workspace/riscv-pk/build/pk}
GCC=${GCC:-/workspace/riscv/bin/riscv64-unknown-elf-gcc}
NM=${NM:-/workspace/riscv/bin/riscv64-unknown-elf-nm}
LANES=${1:-32}
ELF=$HERE/fp8_matmul.elf

python3 "$HERE/ref.py" "$HERE/expected.h" >/dev/null
$GCC -O2 -march=rv64gcv -mabi=lp64d -I"$HERE" "$HERE/fp8_matmul.c" -o "$ELF"

# --kernel-addr is mandatory: outside it spike runs the vector unit with ONE
# lane (decode.h VI_LD) and a lane-banked tile silently comes out 1/L correct.
read -r LO SZ < <($NM -S "$ELF" | awk '$4=="npu_kernel"{print $1, $2}')
HI=$(printf '%x' $((0x$LO + 0x$SZ)))
KRANGE=${KERNEL_ADDR-$LO:$HI}

exec $SPIKE --isa=rv64gcv_zfh_zvfp8 --varch=vlen:256,elen:64 \
  --vectorlane-size=$LANES \
  -m0x80000000:0x80000000,0x70000000:0x$(printf '%x' $((65536 * LANES))) \
  --scratchpad-base-paddr=1879048192 --scratchpad-base-vaddr=3489660928 \
  --scratchpad-size=65536 --kernel-addr=$KRANGE --base-path="$HERE" \
  "$PK" "$ELF" "$LANES" $2
