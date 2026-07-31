// vfsgnj vd, vs2, vs1
VI_VFP_VF_LOOP
({
  vd = fsgnj16(vs2.v, rs1.v, false, false);
},
{
  vd = fsgnj32(vs2.v, rs1.v, false, false);
},
{
  vd = fsgnj64(vs2.v, rs1.v, false, false);
},
{
  // no fsgnj8 macro exists; fp8 sign bit is 0x80
  vd = f8((uint8_t)((vs2.v & 0x7F) | (rs1.v & 0x80)));
})
