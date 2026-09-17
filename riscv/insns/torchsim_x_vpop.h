// The cross-lane unit's pop, once for every combination. THE FIRST POP RUNS THE OP:
// nothing else can, because no lane's answer is known until every lane is in, and
// this is the instruction that says the pushes are done -- WHICH op runs was
// decided by the push, so SIMM5 is not read here.

const reg_t vd = insn.rd();
const reg_t vl = P.VU.vl->read();
const reg_t n_vu = P.VU.get_vu_num();
const reg_t vstart = P.VU.vstart->read();
const char* debug_env = std::getenv("SPIKE_XLU_DEBUG");
const int debug_flag = debug_env ? std::stoi(debug_env) : 0;

// PUSHES PENDING MEANS A NEW TILE, and a new tile replaces the old one whether or
// not the old one was drained -- `run` empties the output side first.
if (P.XLU->has_input())
    P.XLU->run();

for (reg_t vu_idx = 0; vu_idx < n_vu; vu_idx++) {
    P.VU.vstart->write(vstart);
    if (debug_flag && vu_idx < 8) {
        printf("[%s] lane[%ld] ", "X_VPOP", vu_idx);
    }
    for (reg_t i = 0; i < vl; ++i) {
        // A LANE THE OP GAVE NOTHING KEEPS WHAT `vd` HELD, and that is not an
        // error: a transposed tile is only `depth` lanes wide, so the lanes past
        // that were never written.
        if (P.XLU->out_empty(vu_idx))
            break;

        VI_STRIP(i);
        P.VU.vstart->write(i);
        //: THE BITS BACK AS THEY WENT. What a narrower element rode in the low
        //: bits of is what it comes out of; nothing here converts, because nothing
        //: between the push and this read what the bits meant.
        uint32_t val = P.XLU->pop(vu_idx);
        switch (P.VU.vsew) {
          case e8:
            P.VU.elt<uint8_t>(vd, vreg_inx, vu_idx, true) = (uint8_t)val;
            break;
          case e16:
            P.VU.elt<uint16_t>(vd, vreg_inx, vu_idx, true) = (uint16_t)val;
            break;
          case e32:
          default:
            P.VU.elt<uint32_t>(vd, vreg_inx, vu_idx, true) = val;
            break;
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
