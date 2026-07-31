// vfmv_s_f: vd[0] = rs1 (vs2=0)
require_vector(true);
require_fp;
require((P.VU.vsew == e8 && p->extension_enabled(EXT_ZVFP8)) ||
        (P.VU.vsew == e16 && p->extension_enabled(EXT_ZFH)) ||
        (P.VU.vsew == e32 && p->extension_enabled('F')) ||
        (P.VU.vsew == e64 && p->extension_enabled('D')));
require(STATE.frm->read() < 0x5);

reg_t vl = P.VU.vl->read();
const reg_t n_vu = P.VU.get_vu_num();
if (vl > 0 && P.VU.vstart->read() < vl) {
  reg_t rd_num = insn.rd();

  for (reg_t vu_idx=0; vu_idx<n_vu; vu_idx++) {
    switch(P.VU.vsew) {
      case e8:
        P.VU.elt<uint8_t>(rd_num, 0, vu_idx, true) = f8(FRS1).v;
        break;
      case e16:
        P.VU.elt<uint16_t>(rd_num, 0, vu_idx, true) = f16(FRS1).v;
        break;
      case e32:
        P.VU.elt<uint32_t>(rd_num, 0, vu_idx, true) = f32(FRS1).v;
        break;
      case e64:
        if (FLEN == 64)
          P.VU.elt<uint64_t>(rd_num, 0, vu_idx, true) = f64(FRS1).v;
        else
          P.VU.elt<uint64_t>(rd_num, 0, vu_idx, true) = f32(FRS1).v;
        break;
    }
  }
}
P.VU.vstart->write(0);
