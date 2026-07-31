reg_t vd = insn.rd();
// The e8/e16 paths round the fp32 accumulator down, so frm has to be injected
// here as every other rounding site does. Without it f32_to_f8/f16 inherit
// whatever FP instruction ran last, anywhere in the program.
softfloat_roundingMode = STATE.frm->read();
const reg_t vl = P.VU.vl->read();
const uint32_t n_vu = P.VU.get_vu_num();
const reg_t vstart = P.VU.vstart->read();
const char* debug_env = std::getenv("SPIKE_DEBUG");
const int debug_flag = debug_env ? std::stoi(debug_env) : 0;

assert(P.SA->n_output >= vl);
for (reg_t vu_idx=0; vu_idx<n_vu; vu_idx++) {
    P.VU.vstart->write(vstart);
    if (debug_flag) {
        printf("[VPOP] lane[%ld]", vu_idx);
    }
    for (reg_t i = 0; i < vl; ++i) {
        if (P.SA->deserializer[vu_idx]->empty())
            break;

        VI_STRIP(i);
        P.VU.vstart->write(i);
        float val = P.SA->deserializer_pop(vu_idx);
        switch (P.VU.vsew) {
          case e8:
            // Unrecognised dtype keeps meaning int8; see torchsim_i_vpush.h.
            if (P.VU.elem_dtype == ELEM_DTYPE_FP8E4M3 ||
                P.VU.elem_dtype == ELEM_DTYPE_FP8E5M2) {
              softfloat_fp8Format = (P.VU.elem_dtype == ELEM_DTYPE_FP8E5M2)
                                    ? softfloat_fp8_e5m2 : softfloat_fp8_e4m3;
              float32_t fp32;
              memcpy(&fp32.v, &val, sizeof(float));
              P.VU.elt<float8_t>(vd, vreg_inx, vu_idx, true) = f32_to_f8(fp32);
            } else {
              P.VU.elt<int8_t>(vd, vreg_inx, vu_idx, true) = static_cast<int8_t>(val);
            }
            break;
          case e16: {
            float32_t fp32;
            memcpy(&fp32.v, &val, sizeof(float));
            P.VU.elt<float16_t>(vd, vreg_inx, vu_idx, true) = f32_to_f16(fp32);
            break;
          }
          case e32:
            P.VU.elt<float>(vd, vreg_inx, vu_idx, true) = val;
            break;
          default:
            P.VU.elt<float>(vd, vreg_inx, vu_idx, true) = val;
            break;
        }
        if (debug_flag) {
            printf("%f ", val);
        }
    }
    if (debug_flag) {
        printf("\n");
    }
}
P.VU.vstart->write(0);
P.SA->n_output -= vl;