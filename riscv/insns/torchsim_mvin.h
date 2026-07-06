static int movin = 1;
std::map<uint64_t, uint64_t> indirect_map;
#include "torchsim_mvin_common.h"

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