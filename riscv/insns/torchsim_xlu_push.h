// The push half of every cross-lane machine, once. `XLU_PUSH_OP` names which one:
// the operation is chosen at the push because a fold configures an accumulator,
// and `XLU_PUSH_TAG` is only what the debug print calls it.
// IT DOES NOT FIRE THE OP, unlike the systolic array's push -- no lane's answer is
// known until every lane is in, so the pop is what runs it.

const reg_t vs = insn.rs2();
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
        //: THE BITS AS THEY ARE, whatever the SEW says they mean. This unit does
        //: no arithmetic, so there is nothing to convert FOR -- a narrower element
        //: rides the low bits of a word and comes back out of them.
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
        P.XLU->push(vu_idx, val);
        if (debug_flag && vu_idx < 8) {
            printf("0x%08x ", val);
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
