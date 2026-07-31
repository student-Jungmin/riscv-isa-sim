reg_t vs = insn.rs2();
const reg_t vl = P.VU.vl->read();
const reg_t n_vu = P.VU.get_vu_num();
const reg_t vstart = P.VU.vstart->read();
const char* debug_env = std::getenv("SPIKE_DEBUG");
const int debug_flag = debug_env ? std::stoi(debug_env) : 0;

for (reg_t vu_idx=0; vu_idx<n_vu; vu_idx++) {
    P.VU.vstart->write(vstart);
    if (debug_flag) {
        printf("[VPUSH_W] lane[%ld]", vu_idx);
    }
    for (reg_t i = 0; i < vl; ++i) {
        VI_STRIP(i);
        P.VU.vstart->write(i);
        float val;
        switch (P.VU.vsew) {
          case e8:
            // Unrecognised dtype keeps meaning int8; see torchsim_i_vpush.h.
            if (P.VU.elem_dtype == ELEM_DTYPE_FP8E4M3 ||
                P.VU.elem_dtype == ELEM_DTYPE_FP8E5M2) {
              softfloat_fp8Format = (P.VU.elem_dtype == ELEM_DTYPE_FP8E5M2)
                                    ? softfloat_fp8_e5m2 : softfloat_fp8_e4m3;
              float32_t fp32 = f8_to_f32(P.VU.elt<float8_t>(vs, vreg_inx, vu_idx));
              memcpy(&val, &fp32.v, sizeof(float));
            } else {
              val = static_cast<float>(P.VU.elt<int8_t>(vs, vreg_inx, vu_idx));
            }
            break;
          case e16: {
            float16_t fp16 = P.VU.elt<float16_t>(vs, vreg_inx, vu_idx);
            float32_t fp32 = f16_to_f32(fp16);
            memcpy(&val, &fp32.v, sizeof(float));
            break;
          }
          case e32:
            val = P.VU.elt<float>(vs, vreg_inx, vu_idx);
            break;
          default:
            val = 0.0f;
            break;
        }
        P.SA->w_serializer_vpush(vu_idx, val);
        if (debug_flag) {
            printf("%f ", val);
        }
    }
    if (debug_flag) {
        printf("\n");
    }
}
P.VU.vstart->write(0);
P.SA->n_weight += vl;
P.SA->prefill_weight();
