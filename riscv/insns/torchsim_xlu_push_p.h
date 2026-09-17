// The push that carries its PATTERN beside the data, from the `.ivv` form's second
// vector operand. NO ROW OF THE TILE IS A PATTERN: `rpu` has none to skip, and the
// pattern may differ per row, which one read out of row 0 never could.

const reg_t vs = insn.rs2();          // the data
const reg_t vp = insn.rd();           // the pattern -- a lane number per lane
const reg_t vl = P.VU.vl->read();
const reg_t n_vu = P.VU.get_vu_num();
const reg_t vstart = P.VU.vstart->read();
const char* debug_env = std::getenv("SPIKE_XLU_DEBUG");
const int debug_flag = debug_env ? std::stoi(debug_env) : 0;

P.XLU->set_op(XLU_PUSH_OP);

for (reg_t vu_idx = 0; vu_idx < n_vu; vu_idx++) {
    P.VU.vstart->write(vstart);
    if (debug_flag && vu_idx < 8) {
        printf("[%s] lane[%ld] ", XLU_PUSH_TAG, vu_idx);
    }
    for (reg_t i = 0; i < vl; ++i) {
        VI_STRIP(i);
        P.VU.vstart->write(i);
        //: THE BITS AS THEY ARE, whatever the SEW says they mean -- the unit does
        //: no arithmetic. THE PATTERN IS ALWAYS A WHOLE WORD: it is a lane number,
        //: not an element, so it does not ride inside a narrower one.
        uint32_t val;
        switch (P.VU.vsew) {
          case e8:
            val = P.VU.elt<uint8_t>(vs, vreg_inx, vu_idx);
            break;
          case e16:
            val = P.VU.elt<uint16_t>(vs, vreg_inx, vu_idx);
            break;
          case e32:
            val = P.VU.elt<uint32_t>(vs, vreg_inx, vu_idx);
            break;
          default:
            val = 0;
            break;
        }
        uint32_t pat = P.VU.elt<uint32_t>(vp, vreg_inx, vu_idx);
        P.XLU->push_p(vu_idx, val, pat);
        if (debug_flag && vu_idx < 8) {
            printf("0x%08x<-%u ", val, pat);
        }
    }
    if (debug_flag && vu_idx < 8) {
        printf("\n");
    }
}
P.VU.vstart->write(0);
// THE ROW GREW BY `vl`, ONCE AND NOT PER LANE: every lane was pushed the same
// number of values, and `depth` is the tile's other dimension, not a total.
P.XLU->depth += vl;
