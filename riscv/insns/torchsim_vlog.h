// torchsim_vlog

const reg_t vs = insn.rs2();
const reg_t vd = insn.rd();
const reg_t vl = P.VU.vl->read();
const reg_t n_vu = P.VU.get_vu_num();
const reg_t vstart = P.VU.vstart->read();

for (reg_t vu_idx=0; vu_idx<n_vu; vu_idx++) {
    P.VU.vstart->write(vstart);
    for (reg_t i=0; i<vl; i++) {
        VI_STRIP(i);
        P.VU.vstart->write(i);
        if (P.VU.vsew == e16) {
            float16_t fp16 = P.VU.elt<float16_t>(vs, vreg_inx, vu_idx);
            float32_t fp32 = f16_to_f32(fp16);
            float val; memcpy(&val, &fp32.v, sizeof(float));
            float res = logf(val);
            float32_t res32; memcpy(&res32.v, &res, sizeof(float));
            P.VU.elt<float16_t>(vd, vreg_inx, vu_idx, true) = f32_to_f16(res32);
        } else if (P.VU.vsew == e32) {
            float val = P.VU.elt<float>(vs, vreg_inx, vu_idx);
            P.VU.elt<float>(vd, vreg_inx, vu_idx, true) = logf(val);
        } else {
            fprintf(stderr, "[torchsim_vlog] Unsupported vsew=%ld (only e16/e32 supported)\n", P.VU.vsew);
            require(0);
        }
    }
}
P.VU.vstart->write(0);
