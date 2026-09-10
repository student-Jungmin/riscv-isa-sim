// The cross-lane operations, and the queues they run between.
// EVERY OP DRAINS `in` AND FILLS `out`, and what changes between them is only how a
// value's LANE is rewritten. `run` is the one place that says a tile is complete.
#include "cross_lane_unit.h"

#include <cfloat>

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
  op = XLU_TRANSPOSE;
}

void crossLaneUnit_t::run()
{
  bool debug_flag = get_env_flag("SPIKE_XLU_DEBUG");

  if (depth == 0)
    return;

  // THE WHOLE TILE FIRST, THEN THE OP. A lane's row has to be complete before any
  // column or any reduction is, which is why this cannot happen a push at a time the
  // the systolic array's compute does.
  std::vector<std::vector<float> > tile(n_lane);
  for (uint32_t lane = 0; lane < n_lane; lane++) {
    tile[lane].reserve(depth);
    for (reg_t k = 0; k < depth && !in[lane]->empty(); k++) {
      tile[lane].push_back(in[lane]->front());
      in[lane]->pop();
    }
  }

  if (debug_flag) {
    printf("======= XLU op %d =======\n", (int)op);
    printf("-------- Input (%u lanes x %ld deep) --------\n", n_lane, (long)depth);
    for (uint32_t lane = 0; lane < n_lane && lane < 8; lane++) {
      printf("lane[%u] ", lane);
      for (size_t k = 0; k < tile[lane].size() && k < 8; k++)
        printf("%9f ", tile[lane][k]);
      printf("\n");
    }
  }

  // WHAT A POP DID NOT TAKE IS NOT THIS TILE'S. A push covers every lane but only
  // the used ones carry data, so an output column is `n_lane` long while the pop
  // that reads it takes one register -- the rest is the unused lanes' padding. Left
  // in place it comes back as the NEXT tile's first value, which is a wrong answer
  // and not a missing one.
  for (uint32_t k = 0; k < n_lane; k++)
    while (!out[k]->empty())
      out[k]->pop();

  switch (op) {
    case XLU_REDUCE_ADD:
    case XLU_REDUCE_MAX:
    case XLU_REDUCE_MIN:
      reduce(tile, op);
      break;
    case XLU_BROADCAST:
      broadcast(tile);
      break;
    case XLU_PERMUTE:
      permute(tile);
      break;
    case XLU_TRANSPOSE:
    default:
      transpose(tile);
      break;
  }

  if (debug_flag) {
    printf("-------- Output --------\n");
    for (uint32_t lane = 0; lane < n_lane && lane < 8; lane++) {
      printf("lane[%u] ", lane);
      std::queue<float> peek = *out[lane];
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

// Column k becomes lane k's row. ONLY `depth` LANES RECEIVE ANYTHING -- the tile was
// `n_lane` wide and `depth` deep, so transposed it is `depth` wide, and the lanes
// past that keep whatever they held.
void crossLaneUnit_t::transpose(const std::vector<std::vector<float> > &tile)
{
  for (reg_t k = 0; k < depth && k < n_lane; k++)
    for (uint32_t lane = 0; lane < n_lane; lane++)
      out[k]->push(k < tile[lane].size() ? tile[lane][k] : 0.0f);
}

// Reduce ACROSS THE LANES and leave the depth alone: offset k of every lane becomes
// the reduction of offset k over all lanes. THE RESULT LANDS IN EVERY LANE, because a
// reduced tile is read back by lanes that no longer have an axis to tell them apart.
// EVERY LANE COUNTS. A lane the compiler is not using must hold the identity, the
// same contract the systolic array's zero padding already stands on.
void crossLaneUnit_t::reduce(const std::vector<std::vector<float> > &tile,
                             xlu_op_t kind)
{
  for (reg_t k = 0; k < depth; k++) {
    float acc = kind == XLU_REDUCE_MAX ? -FLT_MAX
              : kind == XLU_REDUCE_MIN ?  FLT_MAX : 0.0f;
    for (uint32_t lane = 0; lane < n_lane; lane++) {
      if (k >= tile[lane].size())
        continue;
      float v = tile[lane][k];
      acc = kind == XLU_REDUCE_MAX ? (v > acc ? v : acc)
          : kind == XLU_REDUCE_MIN ? (v < acc ? v : acc) : acc + v;
    }
    for (uint32_t lane = 0; lane < n_lane; lane++)
      out[lane]->push(acc);
  }
}

// Lane 0's row to every lane. THE SOURCE IS LANE 0 BY CONVENTION and not by
// election: a value with no lane axis is the one a single bank holds, and a DMA
// that staged one element staged it there.
// ROW 0 IS THE PATTERN AND NOT DATA: one lane number per lane, saying where that
// lane READS FROM. The tile follows it through the same queue, because an sf.vc form
// carries one vector operand and there is no second one to describe a mapping with.
void crossLaneUnit_t::permute(const std::vector<std::vector<float> > &tile)
{
  for (uint32_t lane = 0; lane < n_lane; lane++) {
    uint32_t from = tile[lane].empty() ? lane : (uint32_t)tile[lane][0];
    for (reg_t k = 1; k < depth; k++) {
      bool have = from < n_lane && k < tile[from].size();
      out[lane]->push(have ? tile[from][k] : 0.0f);
    }
  }
}

void crossLaneUnit_t::broadcast(const std::vector<std::vector<float> > &tile)
{
  for (reg_t k = 0; k < depth; k++) {
    float v = k < tile[0].size() ? tile[0][k] : 0.0f;
    for (uint32_t lane = 0; lane < n_lane; lane++)
      out[lane]->push(v);
  }
}
