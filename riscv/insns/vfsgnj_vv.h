// vfsgnj
VI_VFP_VV_LOOP
({
  vd = fsgnj16(vs2.v, vs1.v, false, false);
},
{
  vd = fsgnj32(vs2.v, vs1.v, false, false);
},
{
  vd = fsgnj64(vs2.v, vs1.v, false, false);
},
{
  // decode.h has no fsgnj8/F8_SIGN, so fsgnj16(a, b, false, false) is expanded
  // here by hand with the fp8 sign bit 0x80: magnitude of vs2, sign of vs1.
  vd = f8((uint8_t)((vs2.v & 0x7F) | (vs1.v & 0x80)));
})
