// The transpose itself, and the queues it runs between.
// `transpose` DRAINS `in` AND FILLS `out`: lane L's row becomes column L, so the
// value at (lane L, offset k) leaves at (lane k, offset L).
#include "cross_lane_unit.h"

void crossLaneUnit_t::reset()
{
  free_queues();

  in = new std::queue<float>*[n_lane];
  out = new std::queue<float>*[n_lane];
  for (uint32_t i = 0; i < n_lane; i++)
    in[i] = new std::queue<float>();
  for (uint32_t i = 0; i < n_lane; i++)
    out[i] = new std::queue<float>();

  depth = 0;
}

void crossLaneUnit_t::transpose()
{
  bool debug_flag = get_env_flag("SPIKE_XLU_DEBUG");

  if (depth == 0)
    return;

  // THE WHOLE TILE FIRST, THEN THE SWAP. A lane's row has to be complete before any
  // column is, which is why this cannot happen a push at a time the way the
  // systolic array's compute does.
  std::vector<std::vector<float>> tile(n_lane);
  for (uint32_t lane = 0; lane < n_lane; lane++) {
    tile[lane].reserve(depth);
    for (reg_t k = 0; k < depth && !in[lane]->empty(); k++) {
      tile[lane].push_back(in[lane]->front());
      in[lane]->pop();
    }
  }

  if (debug_flag) {
    printf("======= TRANSPOSE =======\n");
    printf("-------- Input (%u lanes x %ld deep) --------\n", n_lane, (long)depth);
    for (uint32_t lane = 0; lane < n_lane && lane < 8; lane++) {
      printf("lane[%u] ", lane);
      for (size_t k = 0; k < tile[lane].size() && k < 8; k++)
        printf("%9f ", tile[lane][k]);
      printf("\n");
    }
  }

  // WHAT A POP DID NOT TAKE IS NOT THIS TILE'S. A push covers every lane but only
  // the used ones carry data, so a transposed column is `n_lane` long while the pop
  // that reads it takes one register -- the rest is the unused lanes' padding. Left
  // in place it comes back as the NEXT tile's first column, which is a wrong answer
  // and not a missing one.
  for (uint32_t k = 0; k < n_lane; k++)
    while (!out[k]->empty())
      out[k]->pop();

  // Column k becomes lane k's row. ONLY `depth` LANES RECEIVE ANYTHING -- the tile
  // was `n_lane` wide and `depth` deep, so transposed it is `depth` wide, and the
  // lanes past that keep whatever they held.
  for (reg_t k = 0; k < depth && k < n_lane; k++)
    for (uint32_t lane = 0; lane < n_lane; lane++)
      out[k]->push(k < tile[lane].size() ? tile[lane][k] : 0.0f);

  if (debug_flag) {
    printf("-------- Output (%ld lanes x %u deep) --------\n", (long)depth, n_lane);
    for (reg_t k = 0; k < depth && k < 8; k++) {
      printf("lane[%ld] ", (long)k);
      std::queue<float> peek = *out[k];
      for (uint32_t i = 0; i < 8 && !peek.empty(); i++) {
        printf("%9f ", peek.front());
        peek.pop();
      }
      printf("\n");
    }
    printf("\n");
  }

  depth = 0;
}
