// vfsgnx
VI_VFP_VV_LOOP
({
  vd = fsgnj16(vs2.v, vs1.v, false, true);
},
{
  vd = fsgnj32(vs2.v, vs1.v, false, true);
},
{
  vd = fsgnj64(vs2.v, vs1.v, false, true);
},
{
  // decode.h has no fsgnj8/F8_SIGN, so fsgnj16(a, b, false, true) is expanded
  // here by hand with the fp8 sign bit 0x80: signs of vs2 and vs1 xored.
  vd = f8((uint8_t)((vs2.v & 0x7F) | ((vs2.v ^ vs1.v) & 0x80)));
})
