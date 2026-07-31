// vfcvt.rtz.x.f.v vd, vd2, vm
VI_VFP_VF_LOOP
({
  P.VU.elt<int16_t>(rd_num, i, vu_idx) = f16_to_i16(vs2, softfloat_round_minMag, true);
},
{
  P.VU.elt<int32_t>(rd_num, i, vu_idx) = f32_to_i32(vs2, softfloat_round_minMag, true);
},
{
  P.VU.elt<int64_t>(rd_num, i, vu_idx) = f64_to_i64(vs2, softfloat_round_minMag, true);
},
{
  P.VU.elt<int8_t>(rd_num, i, vu_idx) = f8_to_i8(vs2, softfloat_round_minMag, true);
})
