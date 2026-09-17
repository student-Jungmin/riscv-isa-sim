// The cross-lane unit's push, once for every combination. SIMM5 IS THE OPERATION
// AND IT IS READ HERE, not compiled in: it left the instruction's mask, so one
// encoding carries all fourteen passes and the pattern loads that are not passes.

const reg_t vs = insn.rs2();
const uint32_t simm5 = (uint32_t)(insn.v_simm5() & 0x1f);
const reg_t vl = P.VU.vl->read();
const reg_t n_vu = P.VU.get_vu_num();
const reg_t vstart = P.VU.vstart->read();
const char* debug_env = std::getenv("SPIKE_XLU_DEBUG");
const int debug_flag = debug_env ? std::stoi(debug_env) : 0;
const bool is_load = xlu_is_load(simm5);

// A LOAD IS NOT A PASS: it fills the post stage's pattern and touches neither the
// tile nor `depth` nor the pending op, so the pass that follows is unchanged by it.
if (!is_load)
    P.XLU->set_op(simm5);

for (reg_t vu_idx = 0; vu_idx < n_vu; vu_idx++) {
    P.VU.vstart->write(vstart);
    if (debug_flag && vu_idx < 8) {
        printf("[X_VPUSH simm5=%u%s] lane[%ld] ", simm5,
               is_load ? " load" : "", vu_idx);
    }
    for (reg_t i = 0; i < vl; ++i) {
        VI_STRIP(i);
        P.VU.vstart->write(i);
        //: THE BITS AS THEY ARE, whatever the SEW says they mean. This unit does
        //: no arithmetic, so there is nothing to convert FOR -- a narrower element
        //: rides the low bits of a word and comes back out of them. A LANE NUMBER
        //: IS ALWAYS A WHOLE WORD, which is why a load reads e32 regardless.
        uint32_t val;
        if (is_load) {
            val = P.VU.elt<uint32_t>(vs, vreg_inx, vu_idx);
            P.XLU->load_post(vu_idx, val);
        } else {
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
        }
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
// number of values, and `depth` is the tile's other dimension, not a total. A LOAD
// GROWS NOTHING -- it fills a pattern queue, and the tile is the same tile.
if (!is_load)
    P.XLU->depth += vl;
