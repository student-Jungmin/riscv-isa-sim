// vfwnmacc.vv vd, vs2, vs1
VI_VFP_VV_LOOP_WIDE
({
  vd = f32_mulAdd(f32(vs1.v ^ F32_SIGN), vs2, f32(vd.v ^ F32_SIGN));
},
{
  vd = f64_mulAdd(f64(vs1.v ^ F64_SIGN), vs2, f64(vd.v ^ F64_SIGN));
},
{
  vd = f16_mulAdd(f16(vs1.v ^ F16_SIGN), vs2, f16(vd.v ^ F16_SIGN));
})
