// The push that carries the PRE stage's pattern beside the data, in the `.ivv`
// form's second vector. One entry per value, so this queue is always exactly as
// long as the pre stage is wide -- only the post stage's can ever be mismatched.

const reg_t vs = insn.rs2();          // the data
const reg_t vp = insn.rd();           // the pattern -- a lane number per lane
const uint32_t simm5 = (uint32_t)(insn.v_simm5() & 0x1f);
const reg_t vl = P.VU.vl->read();
const reg_t n_vu = P.VU.get_vu_num();
const reg_t vstart = P.VU.vstart->read();
const char* debug_env = std::getenv("SPIKE_XLU_DEBUG");
const int debug_flag = debug_env ? std::stoi(debug_env) : 0;

P.XLU->set_op(simm5);

for (reg_t vu_idx = 0; vu_idx < n_vu; vu_idx++) {
    P.VU.vstart->write(vstart);
    if (debug_flag && vu_idx < 8) {
        printf("[X_VPUSH_P simm5=%u] lane[%ld] ", simm5, vu_idx);
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
