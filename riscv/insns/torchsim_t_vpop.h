// torchsim_t_vpop -- one vector register out of the transpose unit, a column per lane.
// THE FIRST POP RUNS THE TRANSPOSE. Nothing else can: a column is only known once
// every row is in, and this is the instruction that says the rows are done.

const reg_t vd = insn.rd();
const reg_t vl = P.VU.vl->read();
const reg_t n_vu = P.VU.get_vu_num();
const reg_t vstart = P.VU.vstart->read();
const char* debug_env = std::getenv("SPIKE_XLU_DEBUG");
const int debug_flag = debug_env ? std::stoi(debug_env) : 0;

// PUSHES PENDING MEANS A NEW TILE, and a new tile replaces the old one whether or
// not the old one was drained -- `transpose` empties the output side first.
if (P.XLU->has_input())
    P.XLU->transpose();

for (reg_t vu_idx = 0; vu_idx < n_vu; vu_idx++) {
    P.VU.vstart->write(vstart);
    if (debug_flag && vu_idx < 8) {
        printf("[T_VPOP] lane[%ld] ", vu_idx);
    }
    for (reg_t i = 0; i < vl; ++i) {
        // A LANE PAST THE TRANSPOSED WIDTH HAS NOTHING, and that is not an error:
        // the tile was `n_lane` wide and `depth` deep, so only `depth` lanes hold
        // a column. Those lanes keep whatever `vd` held.
        if (P.XLU->out_empty(vu_idx))
            break;

        VI_STRIP(i);
        P.VU.vstart->write(i);
        float val = P.XLU->pop(vu_idx);
        switch (P.VU.vsew) {
          case e8:
            P.VU.elt<int8_t>(vd, vreg_inx, vu_idx, true) = static_cast<int8_t>(val);
            break;
          case e16: {
            float32_t fp32;
            memcpy(&fp32.v, &val, sizeof(float));
            P.VU.elt<float16_t>(vd, vreg_inx, vu_idx, true) = f32_to_f16(fp32);
            break;
          }
          case e32:
          default:
            P.VU.elt<float>(vd, vreg_inx, vu_idx, true) = val;
            break;
        }
        if (debug_flag && vu_idx < 8) {
            printf("%f ", val);
        }
    }
    if (debug_flag && vu_idx < 8) {
        printf("\n");
    }
}
P.VU.vstart->write(0);
