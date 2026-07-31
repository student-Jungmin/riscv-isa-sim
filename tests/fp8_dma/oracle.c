/* Host cross-check of the hand-derived fp8 tables in fp8_dma.c. */
#include <stdio.h>
#include <stdint.h>
#include "softfloat.h"

static const uint8_t va[8] = {0x38,0x40,0x38,0x30,0xB8,0x01,0x48,0x44};
static const uint8_t vb[8] = {0x38,0x40,0x40,0x38,0x40,0x01,0x30,0x44};

int main(void){
  for (int f=0; f<2; f++) {
    softfloat_fp8Format = f ? softfloat_fp8_e5m2 : softfloat_fp8_e4m3;
    printf("static const uint8_t add_%s[8] = {", f?"e5":"e4");
    for (int i=0;i<8;i++){ float8_t a={va[i]},b={vb[i]};
      softfloat_exceptionFlags=0; printf("0x%02X%s", f8_add(a,b).v, i<7?", ":"");}
    printf("};\n");
    printf("static const uint8_t mul_%s[8] = {", f?"e5":"e4");
    for (int i=0;i<8;i++){ float8_t a={va[i]},b={vb[i]};
      softfloat_exceptionFlags=0; printf("0x%02X%s", f8_mul(a,b).v, i<7?", ":"");}
    printf("};\n");
  }
  return 0;
}
