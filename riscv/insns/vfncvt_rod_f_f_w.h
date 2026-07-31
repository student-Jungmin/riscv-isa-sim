//vfncvt_rod_f_f_w.// vfncvt.rod.f.f.v vd, vs2, vm
VI_VFP_CVT_SCALE
({
  // f16_to_f8 rounds in f32_to_f8, which honours softfloat_round_odd under
  // SOFTFLOAT_ROUND_ODD (defined in platform.h), so rod is real here.
  softfloat_roundingMode = softfloat_round_odd;
  auto vs2 = P.VU.elt<float16_t>(rs2_num, i, vu_idx);
  P.VU.elt<float8_t>(rd_num, i, vu_idx, true) = f16_to_f8(vs2);
},
{
  softfloat_roundingMode = softfloat_round_odd;
  auto vs2 = P.VU.elt<float32_t>(rs2_num, i, vu_idx);
  P.VU.elt<float16_t>(rd_num, i, vu_idx, true) = f32_to_f16(vs2);
},
{
  softfloat_roundingMode = softfloat_round_odd;
  auto vs2 = P.VU.elt<float64_t>(rs2_num, i, vu_idx);
  P.VU.elt<float32_t>(rd_num, i, vu_idx, true) = f64_to_f32(vs2);
},
{
  // The loop base additionally requires Zfh at e8; the f16 source needs it
  // anyway, so nothing is over-gated here.
  // The wide side really is f16 here, so this one needs Zfh as well.
  require(p->extension_enabled(EXT_ZVFP8));
  require(p->extension_enabled(EXT_ZFH));
},
{
  require(p->extension_enabled(EXT_ZFH));
},
{
  require(p->extension_enabled('F'));
},
false, (P.VU.vsew >= 8))
