// vmv_s_x: vd[0] = rs1
require_vector(true);
require(insn.v_vm() == 1);
require(P.VU.vsew >= e8 && P.VU.vsew <= e64);
reg_t vl = P.VU.vl->read();
const reg_t n_vu = P.VU.get_vu_num();
if (vl > 0 && P.VU.vstart->read() < vl) {
  reg_t rd_num = insn.rd();
  reg_t sew = P.VU.vsew;

  for (reg_t vu_idx=0; vu_idx<n_vu; vu_idx++) {
    switch(sew) {
    case e8:
      P.VU.elt<uint8_t>(rd_num, 0, vu_idx, true) = RS1;
      break;
    case e16:
      P.VU.elt<uint16_t>(rd_num, 0, vu_idx, true) = RS1;
      break;
    case e32:
      P.VU.elt<uint32_t>(rd_num, 0, vu_idx, true) = RS1;
      break;
    default:
      P.VU.elt<uint64_t>(rd_num, 0, vu_idx, true) = RS1;
      break;
    }
  }
}
P.VU.vstart->write(0);
