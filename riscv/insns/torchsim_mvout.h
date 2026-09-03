// Custom instruction mvin
// mvin rs1, rs2
// rs1 = virtual main memory address
// rs2: scratchpad address
#include "common.h"

#define N 0
#define C 1
#define H 2
#define W 3
#define ROUNDUP(X, Y) (((X) + (Y) - 1) / (Y) * (Y))
static int indirect_counter;
std::map<uint64_t, uint64_t> indirect_map;

const char* debug_env = std::getenv("SPIKE_DEBUG");
const int debug_flag = debug_env ? std::stoi(debug_env) : 0;

const uint64_t n_vu = P.VU.get_vu_num();

const reg_t dramAddr = RS1;
const reg_t scratchpadAddr = RS2;

// TMA-style descriptor (see project-dma-descriptor); mvout reads the same struct.
reg_t desc_dim_size[4], desc_mm_stride[4], desc_spad_stride[4];
reg_t desc_dim_low[4], desc_dim_high[4];
reg_t desc_element_size, desc_vlane_stride; int desc_vlane_split_axis; bool desc_indirect;
reg_t desc_indirect_addr, desc_indirect_stride, desc_indirect_esize; uint64_t desc_fill;
int desc_indirect_dim; reg_t desc_indirect_lanes;
const reg_t descp = P.VU.dma_desc_ptr;
for (int i = 0; i < 4; i++) {
  desc_dim_size[i]   = (int32_t)MMU.load_uint32(descp + 0  + 4*i);
  desc_dim_low[i]    = (int32_t)MMU.load_uint32(descp + 16 + 4*i);
  desc_dim_high[i]   = (int32_t)MMU.load_uint32(descp + 32 + 4*i);
  desc_mm_stride[i]  = (int64_t)MMU.load_uint64(descp + 48 + 8*i);
  desc_spad_stride[i]= (int64_t)MMU.load_uint64(descp + 80 + 8*i);
}
desc_element_size     = MMU.load_uint16(descp + 112);
desc_vlane_stride     = MMU.load_uint16(descp + 114);
desc_vlane_split_axis = MMU.load_uint8(descp + 116);
uint16_t desc_flags   = MMU.load_uint16(descp + 118);
desc_indirect         = desc_flags & 0x1;
const bool desc_masked = desc_flags & 0x2;
const bool desc_accumulate = desc_flags & 0x4;   // index_add: out[d_addr] += val
const bool desc_acc_float  = desc_flags & 0x8;   // accumulate with float (else integer) add
desc_indirect_addr    = MMU.load_uint64(descp + 120);
desc_indirect_stride  = MMU.load_uint16(descp + 128);
desc_indirect_esize   = MMU.load_uint16(descp + 130);
desc_indirect_dim     = MMU.load_uint8(descp + 132);
desc_indirect_lanes   = MMU.load_uint16(descp + 134);
desc_fill             = MMU.load_uint64(descp + 136);
// THE DRAM RANGE THIS TRANSFER MAY TOUCH, or 0 for "no bound". An address the
// walk or an indirect index puts outside it is NOT this transfer's to write --
// which is how a masked-off element is sent nowhere instead of somewhere.
const reg_t desc_dram_base  = MMU.load_uint64(descp + 144);
const reg_t desc_dram_bytes = MMU.load_uint64(descp + 152);

const reg_t *p_dim_size = desc_dim_size;
const reg_t *p_mm_stride = desc_mm_stride;
const reg_t *p_spad_stride = desc_spad_stride;
const reg_t element_size = desc_element_size;
const reg_t vlane_stride = desc_vlane_stride;
const int vlane_split_axis = desc_vlane_split_axis;
const bool indirect_mode = desc_indirect;
uint64_t n_outerloop = 1;

if (vlane_split_axis == N)
    n_outerloop = (p_dim_size[0] + (vlane_stride * n_vu) - 1) / (vlane_stride * n_vu);
else if (vlane_split_axis == C)
    n_outerloop = (p_dim_size[1] + (vlane_stride * n_vu) - 1) / (vlane_stride * n_vu);
else if (vlane_split_axis == H)
    n_outerloop = (p_dim_size[2] + (vlane_stride * n_vu) - 1) / (vlane_stride * n_vu);
else if (vlane_split_axis == W)
    n_outerloop = (p_dim_size[3] + (vlane_stride * n_vu) - 1) / (vlane_stride * n_vu);
else
    assert(0);

uint64_t used_vlane = n_outerloop > 1 ? n_vu : (p_dim_size[vlane_split_axis] + vlane_stride - 1) / vlane_stride;

uint64_t block_dim[4] = {p_dim_size[N], p_dim_size[C], p_dim_size[H], p_dim_size[W]};
block_dim[vlane_split_axis] = vlane_stride;

uint64_t block_stride[4] = {p_spad_stride[N], p_spad_stride[C], p_spad_stride[H], p_spad_stride[W]};
for (int i=0; i<4; i++) {
    if (block_stride[i] > p_spad_stride[vlane_split_axis])
        block_stride[i] = (block_stride[i] / p_dim_size[vlane_split_axis]) * vlane_stride * n_outerloop;
}

uint64_t buffer_size = ROUNDUP(p_dim_size[0] * p_dim_size[1] * p_dim_size[2] * p_dim_size[3], used_vlane * block_dim[N] * block_dim[C] * block_dim[H] * block_dim[W]);
uint64_t dma_buffer_stride[4] = {p_dim_size[C] * p_dim_size[H] * p_dim_size[W], p_dim_size[H] * p_dim_size[W], p_dim_size[W], 1};

uint64_t d_vlane_idx_stride = dma_buffer_stride[vlane_split_axis] * vlane_stride;
uint64_t d_outerloop_idx_stride = d_vlane_idx_stride * used_vlane;
uint64_t s_outerloop_idx_stride = block_stride[vlane_split_axis] * vlane_stride;

void *dma_buffer = nullptr;
try {
    dma_buffer = new uint64_t[buffer_size]();   // zero-init: ROUNDUP padding entries stay 0 -> skipped
} catch (const std::bad_alloc& e) {
    std::cerr << "Memory allocation failed: " << e.what() << std::endl;
    assert(false);
}

if (debug_flag) {
    printf("=============== MVOUT ===============\n");
    printf("Instruction configs:\n");
    printf("- dramAddr: 0x%lx\n", dramAddr);
    printf("- scratchpadAddr: 0x%lx\n", scratchpadAddr);
    printf("- indirect mode: %d\n", indirect_mode);
    printf("- p_dim_size: (%ld, %ld, %ld, %ld)\n", p_dim_size[0], p_dim_size[1], p_dim_size[2], p_dim_size[3]);
    printf("- p_mm_stride: (%ld, %ld, %ld, %ld)\n", p_mm_stride[0], p_mm_stride[1], p_mm_stride[2], p_mm_stride[3]);
    printf("- p_spad_stride: (%ld, %ld, %ld, %ld)\n", p_spad_stride[0], p_spad_stride[1], p_spad_stride[2], p_spad_stride[3]);
    printf("- element_size: %ld\n", element_size);
    printf("- vlane_stride: %ld\n", vlane_stride);
    printf("- vlane_split_axis: %d\n", vlane_split_axis);
    printf("- Outer loop: %ld\n", n_outerloop);

    printf("\nDMA buffer configs:\n");
    printf("- dma buffer size: %ld\n", buffer_size);
    printf("- dma buffer stride: (%ld, %ld, %ld, %ld)\n", dma_buffer_stride[0], dma_buffer_stride[1], dma_buffer_stride[2], dma_buffer_stride[3]);

    printf("\nBlock configs:\n");
    printf("- block_dim: (%ld, %ld, %ld, %ld)\n", block_dim[N], block_dim[C], block_dim[H], block_dim[W]);
    printf("- block_stride: (%ld, %ld, %ld, %ld)\n", block_stride[N], block_stride[C], block_stride[H], block_stride[W]);
    printf("- used_vlane: %ld\n", used_vlane);
    printf("- d_vlane_idx_stride: %ld\n", d_vlane_idx_stride);
    printf("- d_outerloop_idx_stride: %ld\n", d_outerloop_idx_stride);
    printf("- s_outerloop_idx_stride: %ld\n", s_outerloop_idx_stride);
}

assert(element_size > 0);
assert(vlane_stride > 0);

// Record target dram address in dma_buffer
for (uint64_t n=0; n<p_dim_size[0]; n++) {
    for (uint64_t c=0; c<p_dim_size[1]; c++) {
        for (uint64_t h=0; h<p_dim_size[2]; h++) {
            for (uint64_t w=0; w<p_dim_size[3]; w++) {
                uint64_t d_offset = (n * p_mm_stride[0] + c * p_mm_stride[1] + h * p_mm_stride[2] + w * p_mm_stride[3]) * element_size;
                uint64_t d_addr = dramAddr + d_offset;
                uint64_t buffer_idx = n * p_dim_size[1] * p_dim_size[2] * p_dim_size[3] + c * p_dim_size[2] * p_dim_size[3] + h * p_dim_size[3] + w;
                // masked-DMA tail clamp: positions >= dim_high are the lane-align tail;
                // sentinel 0 makes the store loop skip them (no OOB write to DRAM).
                bool in_box = !desc_masked || (
                    (int64_t)n >= desc_dim_low[0] && (int64_t)n < desc_dim_high[0] &&
                    (int64_t)c >= desc_dim_low[1] && (int64_t)c < desc_dim_high[1] &&
                    (int64_t)h >= desc_dim_low[2] && (int64_t)h < desc_dim_high[2] &&
                    (int64_t)w >= desc_dim_low[3] && (int64_t)w < desc_dim_high[3]);
                static_cast<uint64_t*>(dma_buffer)[buffer_idx] = in_box ? d_addr : 0;
            }
        }
    }
}

for (uint64_t outerloop_idx=0; outerloop_idx<n_outerloop; outerloop_idx++) {
    for (uint64_t vlane_idx=0; vlane_idx<used_vlane; vlane_idx++) {
        for (uint64_t n=0; n<block_dim[N]; n++) {
            for (uint64_t c=0; c<block_dim[C]; c++) {
                for (uint64_t h=0; h<block_dim[H]; h++) {
                    for (uint64_t w=0; w<block_dim[W]; w++) {
                        uint64_t d_idx = d_outerloop_idx_stride * outerloop_idx + d_vlane_idx_stride * vlane_idx + dma_buffer_stride[N] * n + dma_buffer_stride[C] * c + dma_buffer_stride[H] * h + dma_buffer_stride[W] * w ;
                        uint64_t s_idx = (s_outerloop_idx_stride * outerloop_idx + block_stride[N] * n + block_stride[C] * c + block_stride[H] * h + block_stride[W] * w);
                        uint64_t s_addr = scratchpadAddr + s_idx * element_size + vlane_idx * P.VU.vu_sram_byte;
                        uint64_t d_addr = static_cast<uint64_t*>(dma_buffer)[d_idx];

                        /* Skip if the dram_addr is not defined */
                        if (d_addr == 0)
                            continue;

                        if (scratchpadAddr + s_idx * element_size >= P.VU.sram_v_space.first + P.VU.vu_sram_byte) {
                            fprintf(stderr, "MVOUT ERROR: Scratchpad address overflow: 0x%lx\n", s_addr);
                            exit(INVALID_SPAD_ACCESS);
                        }

                        if (debug_flag)
                            printf("[MOVOUT] outerloop_idx: %ld, vlane_idx: %ld, N: %ld, C: %ld, H: %ld, W: %ld\n", outerloop_idx, vlane_idx, n, c, h, w);

                        if (indirect_mode) {
                            uint64_t indirect_base_addr = desc_indirect_addr;
                            uint64_t indirect_stride = desc_indirect_stride;
                            uint64_t indirect_element_size = desc_indirect_esize;
                            // ONE INDEX PER AXIS POSITION, NOT PER ELEMENT. dim k+1
                            // says the index varies along axis k alone, so it is read
                            // at that axis's GLOBAL coordinate; 0 keeps the old walk.
                            uint64_t indirect_addr;
                            if (desc_indirect_dim) {
                                const int ia = desc_indirect_dim - 1;
                                uint64_t i_crd = (ia == N) ? n : (ia == C) ? c : (ia == H) ? h : w;
                                if (ia == vlane_split_axis)
                                    i_crd += (outerloop_idx * used_vlane + vlane_idx) * vlane_stride;
                                // THE INDEX IS ITS OWN RANK-1 TILE: entry p sits in lane
                                // p % lanes at offset p / lanes, and ONE lane is the
                                // legal answer too -- so how many lanes hold it is the
                                // compiler's to say, not this loop's to guess.
                                uint64_t i_lanes = desc_indirect_lanes ? desc_indirect_lanes : 1;
                                indirect_addr = indirect_base_addr
                                              + (i_crd / i_lanes) * indirect_element_size
                                              + (i_crd % i_lanes) * P.VU.vu_sram_byte;
                            } else {
                                indirect_addr = indirect_base_addr + s_idx * indirect_element_size
                                              + vlane_idx * P.VU.vu_sram_byte;
                            }
                            uint64_t indirect_idx = 0;
                            // AN INDEX IS SIGNED, AND THE FIELD IT ARRIVES IN IS
                            // NARROWER THAN AN ADDRESS. Read unsigned, -128 in a
                            // four-byte field becomes 4294967168 and
                            // base + idx * scale lands 1 << (32 + log2(scale))
                            // past the tile -- measured as a User load segfault
                            // on ms_deform_attn and on masked_negative_offset,
                            // whose kernel computed idx - 128. At eight bytes the
                            // zero extension is the identity, so the wrap in
                            // two's complement saved it and only the narrow
                            // fields were ever wrong. Sign-extending here is what
                            // that width was missing; the compiler no longer has
                            // to widen every index tile to i64 to dodge it.
                            switch(indirect_element_size) {
                            case 1:
                                indirect_idx = (int64_t)MMU.load_int8(indirect_addr);
                                break;
                            case 2:
                                indirect_idx = (int64_t)MMU.load_int16(indirect_addr);
                                break;
                            case 4:
                                indirect_idx = (int64_t)MMU.load_int32(indirect_addr);
                                break;
                            case 8:
                                indirect_idx = MMU.load_uint64(indirect_addr);
                                break;
                            default:
                                fprintf(stderr, "Unsupported index type\n");
                                assert(false);
                            }
                            if (debug_flag) {
                                printf("[Indirect index] Base : 0x%lx, stride: %ld, element_size: %ld, idx: %ld\n",
                                        indirect_base_addr, indirect_stride, indirect_element_size, indirect_idx);
                            }
                            indirect_map[d_idx] = indirect_idx;
                            d_addr += indirect_idx * indirect_stride * element_size;
                        }

                        // OUTSIDE THE ARGUMENT: not ours to write. THE WHOLE POINT
                        // -- a masked-off element is sent here instead of to a
                        // harmless address the wrapper had to reserve for it.
                        if (desc_dram_bytes && (d_addr < desc_dram_base
                                || d_addr + element_size > desc_dram_base + desc_dram_bytes))
                            continue;

                        if (element_size == 1) {
                            uint8_t val = MMU.load_uint8(s_addr);
                            if (desc_accumulate) val = (uint8_t)(MMU.load_uint8(d_addr) + val);
                            MMU.store_uint8(d_addr, val);
                            if (debug_flag) {
                                int8_t as_int = *((int8_t*)&val);
                                printf("- Buffer_idx: %ld, Dram_addr: 0x%lx, Spad_addr: 0x%lx, Val(hex): 0x%02x, Val(int): %d\n",
                                       d_idx, d_addr, s_addr, (uint8_t)val, as_int);
                            }
                        } else if (element_size == 2) {
                            uint16_t val = MMU.load_uint16(s_addr);
                            if (desc_accumulate) {
                                uint16_t cur = MMU.load_uint16(d_addr);
                                if (desc_acc_float) {
                                    float16_t a; a.v = cur; float16_t b; b.v = val;
                                    val = f16_add(a, b).v;
                                } else val = (uint16_t)(cur + val);
                            }
                            MMU.store_uint16(d_addr, val);
                            if (debug_flag) {
                                int16_t as_int = *((int16_t*)&val);
                                printf("- Buffer_idx: %ld, Dram_addr: 0x%lx, Spad_addr: 0x%lx, Val(hex): 0x%04x, Val(int): %d\n",
                                       d_idx, d_addr, s_addr, (uint16_t)val, as_int);
                            }
                        } else if (element_size == 4) {
                            uint32_t val = MMU.load_uint32(s_addr);
                            if (desc_accumulate) {
                                uint32_t cur = MMU.load_uint32(d_addr);
                                if (desc_acc_float) {
                                    float r = *((float*)&cur) + *((float*)&val);
                                    val = *((uint32_t*)&r);
                                } else val = cur + val;
                            }
                            MMU.store_uint32(d_addr, val);
                            if (debug_flag) {
                                int32_t as_int = *((int32_t*)&val);
                                float   as_f32 = *((float*)&val);
                                printf("- Buffer_idx: %ld, Dram_addr: 0x%lx, Spad_addr: 0x%lx, Val(hex): 0x%08x, Val(int): %d, Val(float): %f\n",
                                       d_idx, d_addr, s_addr, val, as_int, as_f32);
                            }
                        } else if (element_size == 8) {
                            uint64_t val = MMU.load_uint64(s_addr);
                            if (desc_accumulate) {
                                uint64_t cur = MMU.load_uint64(d_addr);
                                if (desc_acc_float) {
                                    double r = *((double*)&cur) + *((double*)&val);
                                    val = *((uint64_t*)&r);
                                } else val = cur + val;
                            }
                            MMU.store_uint64(d_addr, val);
                            if (debug_flag) {
                                int64_t as_int = *((int64_t*)&val);
                                double  as_f64 = *((double*)&val);
                                printf("- Buffer_idx: %ld, Dram_addr: 0x%lx, Spad_addr: 0x%lx, Val(hex): 0x%016lx, Val(int): %ld, Val(double): %f\n",
                                       d_idx, d_addr, s_addr, val, as_int, as_f64);
                            }
                        }
                    }
                }
            }
        }
    }
}

if (dma_buffer != nullptr) {
    delete [] static_cast<uint64_t*>(dma_buffer);
    dma_buffer = nullptr;
}

if (desc_indirect) {
    std::string file_path = std::string(P.base_path) + "/indirect_access/indirect_index" + std::to_string(P.VU.dma_indirect_counter++) + ".raw";
    FILE* fp = fopen(file_path.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file for writing: %s\n", file_path);
    } else {
        for (size_t i = 0; i < indirect_map.size(); ++i) {
            fwrite(&(indirect_map[i]), sizeof(uint64_t), 1, fp);
        }
        fclose(fp);
    }
}