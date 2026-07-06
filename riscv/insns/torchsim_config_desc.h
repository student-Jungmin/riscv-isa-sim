// CONFIG_DESC: set the TMA-style DMA descriptor base address.
// rs1 = descriptor struct address (the frontend's memref.global). MVIN/MVOUT read
// the struct from here at exec time, so this replaces CONFIG/CONFIG2/CONFIG3/CONFIG4.
P.VU.dma_desc_ptr = RS1;
