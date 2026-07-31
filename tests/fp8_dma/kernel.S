/* Kernel-PC-range routines. Spike only engages the vector lanes while the PC
   is inside --kernel-addr, so every DMA and vector op the test measures lives
   between k_begin and k_end. */

	.text
	.balign 4
	.globl	k_begin
k_begin:

/* First insn in the range must be an addi/c.lui with a NEGATIVE immediate:
   execute.cc asserts on that exact PC to capture the kernel stack bounds. */
	.globl	k_config
	.type	k_config, @function
k_config:
	addi	sp, sp, -16
	.insn	r 0x2b, 3, 7, x0, a0, x0	/* TORCHSIM_CONFIG_DESC a0 */
	addi	sp, sp, 16
	ret
	.size	k_config, .-k_config

/* Every routine below is frameless on purpose: only the entry above should
   move the tracked kernel sp. */
	.globl	k_mvin
	.type	k_mvin, @function
k_mvin:
	.insn	r 0x2b, 3, 2, x0, a0, a1	/* TORCHSIM_MVIN dram=a0 spad=a1 */
	ret
	.size	k_mvin, .-k_mvin

	.globl	k_mvout
	.type	k_mvout, @function
k_mvout:
	.insn	r 0x2b, 3, 3, x0, a0, a1	/* TORCHSIM_MVOUT dram=a0 spad=a1 */
	ret
	.size	k_mvout, .-k_mvout

	.globl	k_vfadd8
	.type	k_vfadd8, @function
k_vfadd8:
	vsetvli	t0, a3, e8, m1, ta, ma
	vle8.v	v1, (a0)
	vle8.v	v2, (a1)
	vfadd.vv	v3, v1, v2
	vse8.v	v3, (a2)
	ret
	.size	k_vfadd8, .-k_vfadd8

	.globl	k_vfmul8
	.type	k_vfmul8, @function
k_vfmul8:
	vsetvli	t0, a3, e8, m1, ta, ma
	vle8.v	v1, (a0)
	vle8.v	v2, (a1)
	vfmul.vv	v3, v1, v2
	vse8.v	v3, (a2)
	ret
	.size	k_vfmul8, .-k_vfmul8

	.globl	k_vadd8
	.type	k_vadd8, @function
k_vadd8:
	vsetvli	t0, a3, e8, m1, ta, ma
	vle8.v	v1, (a0)
	vle8.v	v2, (a1)
	vadd.vv	v3, v1, v2
	vse8.v	v3, (a2)
	ret
	.size	k_vadd8, .-k_vadd8

	.globl	k_end
k_end:
	.balign	4
