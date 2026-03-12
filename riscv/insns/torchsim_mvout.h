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
const reg_t *p_dim_size = P.VU.dma_dim_size;
const reg_t *p_mm_stride = P.VU.dma_mm_stride;
const reg_t *p_spad_stride = P.VU.dma_spad_stride;
const reg_t element_size = P.VU.dma_element_size;
const reg_t vlane_stride = P.VU.dma_vlane_stride;
const int vlane_split_axis = P.VU.dma_vlane_split_axis;
const bool indirect_mode = P.VU.dma_indirect_mode;
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
    dma_buffer = new uint64_t[buffer_size];
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
                static_cast<uint64_t*>(dma_buffer)[buffer_idx] = d_addr;
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

                        if (scratchpadAddr + s_idx * element_size >= P.VU.sram_v_space.first + P.VU.vu_sram_byte) {
                            fprintf(stderr, "MVOUT ERROR: Scratchpad address overflow: 0x%lx\n", s_addr);
                            exit(INVALID_SPAD_ACCESS);
                        }

                        if (debug_flag)
                            printf("[MOVOUT] outerloop_idx: %ld, vlane_idx: %ld, N: %ld, C: %ld, H: %ld, W: %ld\n", outerloop_idx, vlane_idx, n, c, h, w);

                        if (indirect_mode) {
                            uint64_t indirect_base_addr = P.VU.dma_indirect_addr;
                            uint64_t indirect_stride = P.VU.dma_indirect_stride;
                            uint64_t indirect_element_size = P.VU.dma_indirect_element_size;
                            uint64_t indirect_addr = indirect_base_addr + s_idx * indirect_element_size + vlane_idx * P.VU.vu_sram_byte;
                            uint64_t indirect_idx = 0;
                            switch(indirect_element_size) {
                            case 1:
                                indirect_idx = MMU.load_uint8(indirect_addr);
                                break;
                            case 2:
                                indirect_idx = MMU.load_uint16(indirect_addr);
                                break;
                            case 4:
                                indirect_idx = MMU.load_uint32(indirect_addr);
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

                        if (element_size == 1) {
                            uint8_t val = MMU.load_uint8(s_addr);
                            MMU.store_uint8(d_addr, val);
                            if (debug_flag) {
                                int8_t as_int = *((int8_t*)&val);
                                printf("- Buffer_idx: %ld, Dram_addr: 0x%lx, Spad_addr: 0x%lx, Val(hex): 0x%02x, Val(int): %d\n",
                                       d_idx, d_addr, s_addr, (uint8_t)val, as_int);
                            }
                        } else if (element_size == 2) {
                            uint16_t val = MMU.load_uint16(s_addr);
                            MMU.store_uint16(d_addr, val);
                            if (debug_flag) {
                                int16_t as_int = *((int16_t*)&val);
                                printf("- Buffer_idx: %ld, Dram_addr: 0x%lx, Spad_addr: 0x%lx, Val(hex): 0x%04x, Val(int): %d\n",
                                       d_idx, d_addr, s_addr, (uint16_t)val, as_int);
                            }
                        } else if (element_size == 4) {
                            uint32_t val = MMU.load_uint32(s_addr);
                            MMU.store_uint32(d_addr, val);
                            if (debug_flag) {
                                int32_t as_int = *((int32_t*)&val);
                                float   as_f32 = *((float*)&val);
                                printf("- Buffer_idx: %ld, Dram_addr: 0x%lx, Spad_addr: 0x%lx, Val(hex): 0x%08x, Val(int): %d, Val(float): %f\n",
                                       d_idx, d_addr, s_addr, val, as_int, as_f32);
                            }
                        } else if (element_size == 8) {
                            uint64_t val = MMU.load_uint64(s_addr);
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

if (P.VU.dma_indirect_mode) {
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