// vfwcvt.x.f.v vd, vs2, vm
VI_VFP_CVT_SCALE
({
  // vsew names the narrow side, so e8 is an fp8 source paired with the
  // 2*SEW=16-bit signed integer destination.
  auto vs2 = P.VU.elt<float8_t>(rs2_num, i, vu_idx);
  P.VU.elt<int16_t>(rd_num, i, vu_idx, true) = f8_to_i16(vs2, STATE.frm->read(), true);
},
{
  auto vs2 = P.VU.elt<float16_t>(rs2_num, i, vu_idx);
  P.VU.elt<int32_t>(rd_num, i, vu_idx, true) = f16_to_i32(vs2, STATE.frm->read(), true);
},
{
  auto vs2 = P.VU.elt<float32_t>(rs2_num, i, vu_idx);
  P.VU.elt<int64_t>(rd_num, i, vu_idx, true) = f32_to_i64(vs2, STATE.frm->read(), true);
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
true, (P.VU.vsew >= 8))
