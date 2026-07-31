// CONFIG_DESC: set the TMA-style DMA descriptor base address.
// rs1 = descriptor struct address (the frontend's memref.global). MVIN/MVOUT read
// the struct from here at exec time, so this replaces CONFIG/CONFIG2/CONFIG3/CONFIG4.
P.VU.dma_desc_ptr = RS1;
// Byte 117 was descriptor padding; it now carries enum ELEM_DTYPE so the SA
// boundary can tell int8 from e4m3 from e5m2 at SEW=8. Unset reads as int8.
P.VU.elem_dtype = MMU.load_uint8(RS1 + 117);
