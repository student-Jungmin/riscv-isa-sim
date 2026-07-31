// vfncvt.f.x.v vd, vs2, vm
VI_VFP_CVT_SCALE
({
  // vsew names the narrow side, so e8 is an fp8 destination paired with the
  // 2*SEW=16-bit signed integer source; i32_to_f8 takes the widened value.
  auto vs2 = P.VU.elt<int16_t>(rs2_num, i, vu_idx);
  P.VU.elt<float8_t>(rd_num, i, vu_idx, true) = i32_to_f8(vs2);
},
{
  auto vs2 = P.VU.elt<int32_t>(rs2_num, i, vu_idx);
  P.VU.elt<float16_t>(rd_num, i, vu_idx, true) = i32_to_f16(vs2);
},
{
  auto vs2 = P.VU.elt<int64_t>(rs2_num, i, vu_idx);
  P.VU.elt<float32_t>(rd_num, i, vu_idx, true) = i64_to_f32(vs2);
},
{
  // No f16 is involved, but the loop base also requires Zfh at e8; removing
  // that would mean editing decode.h, so this path is Zfh-gated too.
  require(p->extension_enabled(EXT_ZVFP8));
},
{
  require(p->extension_enabled(EXT_ZFH));
},
{
  require(p->extension_enabled('F'));
},
false, (P.VU.vsew >= 8))
