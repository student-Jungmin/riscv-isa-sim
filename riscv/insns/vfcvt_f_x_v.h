// vfcvt.f.x.v vd, vd2, vm
VI_VFP_VF_LOOP
({
  auto vs2_i = P.VU.elt<int16_t>(rs2_num, i, vu_idx);
  vd = i32_to_f16(vs2_i);
},
{
  auto vs2_i = P.VU.elt<int32_t>(rs2_num, i, vu_idx);
  vd = i32_to_f32(vs2_i);
},
{
  auto vs2_i = P.VU.elt<int64_t>(rs2_num, i, vu_idx);
  vd = i64_to_f64(vs2_i);
},
{
  auto vs2_i = P.VU.elt<int8_t>(rs2_num, i, vu_idx);
  vd = i32_to_f8(vs2_i);
})
