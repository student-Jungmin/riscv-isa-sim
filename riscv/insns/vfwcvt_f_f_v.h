// vfwcvt.f.f.v vd, vs2, vm
VI_VFP_CVT_SCALE
({
  // vsew names the narrow side, so e8 is an fp8 source here, not an integer.
  // f8_to_f16 is exact for both fp8 formats and raises no flag.
  auto vs2 = P.VU.elt<float8_t>(rs2_num, i, vu_idx);
  P.VU.elt<float16_t>(rd_num, i, vu_idx, true) = f8_to_f16(vs2);
},
{
  auto vs2 = P.VU.elt<float16_t>(rs2_num, i, vu_idx);
  P.VU.elt<float32_t>(rd_num, i, vu_idx, true) = f16_to_f32(vs2);
},
{
  auto vs2 = P.VU.elt<float32_t>(rs2_num, i, vu_idx);
  P.VU.elt<float64_t>(rd_num, i, vu_idx, true) = f32_to_f64(vs2);
},
{
  // The loop base additionally requires Zfh at e8; the f16 destination
  // needs it anyway, so nothing is over-gated here.
  // The wide side really is f16 here, so this one needs Zfh as well.
  require(p->extension_enabled(EXT_ZVFP8));
  require(p->extension_enabled(EXT_ZFH));
},
{
  require(p->extension_enabled(EXT_ZFH));
},
{
  require(p->extension_enabled('D'));
},
true, (P.VU.vsew >= 8))
