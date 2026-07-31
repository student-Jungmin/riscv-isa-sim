// vfncvt.f.f.v vd, vs2, vm
VI_VFP_CVT_SCALE
({
  // vsew names the narrow side, so e8 is an fp8 destination, not an integer.
  // This is the only fp8 convert that rounds, so inexact/overflow/underflow
  // relative to fp8 arise here.
  auto vs2 = P.VU.elt<float16_t>(rs2_num, i, vu_idx);
  P.VU.elt<float8_t>(rd_num, i, vu_idx, true) = f16_to_f8(vs2);
},
{
  auto vs2 = P.VU.elt<float32_t>(rs2_num, i, vu_idx);
  P.VU.elt<float16_t>(rd_num, i, vu_idx, true) = f32_to_f16(vs2);
},
{
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
  require(p->extension_enabled('D'));
},
false, (P.VU.vsew >= 8))
