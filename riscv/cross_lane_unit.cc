// The cross-lane operations, and the queues they run between.
// EVERY OP DRAINS `in` AND FILLS `out`, and what changes between them is only how a
// value's LANE is rewritten. `run` is the one place that says a tile is complete.
#include "cross_lane_unit.h"

void crossLaneUnit_t::reset()
{
  free_queues();

  in = new std::queue<uint32_t>*[n_lane];
  out = new std::queue<uint32_t>*[n_lane];
  pre_pat = new std::queue<uint32_t>*[n_lane];
  post_pat = new std::queue<uint32_t>*[n_lane];
  for (uint32_t i = 0; i < n_lane; i++) {
    in[i] = new std::queue<uint32_t>();
    out[i] = new std::queue<uint32_t>();
    pre_pat[i] = new std::queue<uint32_t>();
    post_pat[i] = new std::queue<uint32_t>();
  }

  depth = 0;
  op = 0;
}

void crossLaneUnit_t::run()
{
  bool debug_flag = get_env_flag("SPIKE_XLU_DEBUG");

  if (depth == 0)
    return;

  // THE WHOLE TILE FIRST, THEN THE OP. A lane's row has to be complete before any
  // column or any reduction is, which is why this cannot happen a push at a time the
  // the systolic array's compute does.
  std::vector<std::vector<uint32_t> > tile(n_lane);
  std::vector<std::vector<uint32_t> > pre_p(n_lane), post_p(n_lane);
  for (uint32_t lane = 0; lane < n_lane; lane++) {
    tile[lane].reserve(depth);
    for (reg_t k = 0; k < depth && !in[lane]->empty(); k++) {
      tile[lane].push_back(in[lane]->front());
      in[lane]->pop();
    }
    //: A STAGE'S PATTERN COMES OFF ITS OWN QUEUE, and both drain whether or not
    //: this op reads them -- what one pass did not use is not the next one's.
    while (!pre_pat[lane]->empty()) {
      pre_p[lane].push_back(pre_pat[lane]->front());
      pre_pat[lane]->pop();
    }
    while (!post_pat[lane]->empty()) {
      post_p[lane].push_back(post_pat[lane]->front());
      post_pat[lane]->pop();
    }
  }

  if (debug_flag) {
    printf("======= XLU SIMM5 %u =======\n", op);
    printf("-------- Input (%u lanes x %ld deep) --------\n", n_lane, (long)depth);
    for (uint32_t lane = 0; lane < n_lane && lane < 8; lane++) {
      printf("lane[%u] ", lane);
      for (size_t k = 0; k < tile[lane].size() && k < 8; k++)
        printf("0x%08x ", tile[lane][k]);
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

  // THE THREE STAGES, IN THE ORDER THE DATA MEETS THEM. Each hands a tile to the
  // next, so a combination costs one pass where a flat enumeration cost two -- an
  // all-gather used to leave the unit after the crossing and come back for the
  // replicate, through a vector register both ways.
  tile_t got = rpu(xlu_pre(op), tile, pre_p, "pre");
  if (xlu_xu(op))
    got = crossing(got);
  got = rpu(xlu_post(op), got, post_p, "post");
  emit(got);

  if (debug_flag) {
    printf("-------- Output --------\n");
    for (uint32_t lane = 0; lane < n_lane && lane < 8; lane++) {
      printf("lane[%u] ", lane);
      std::queue<uint32_t> peek = *out[lane];
      for (uint32_t i = 0; i < 8 && !peek.empty(); i++) {
        printf("0x%08x ", peek.front());
        peek.pop();
      }
      printf("\n");
    }
    printf("\n");
  }

  depth = 0;
}

// Column k becomes lane k's row. ONLY `depth` LANES RECEIVE ANYTHING -- the tile was
// `n_lane` wide and `depth` deep, so crossed it is `depth` wide, and the lanes past
// that keep whatever they held.
crossLaneUnit_t::tile_t crossLaneUnit_t::crossing(const tile_t &tile)
{
  tile_t got(n_lane);
  for (reg_t k = 0; k < depth && k < n_lane; k++)
    for (uint32_t lane = 0; lane < n_lane; lane++)
      got[k].push_back(k < tile[lane].size() ? tile[lane][k] : 0u);
  return got;
}

// The crossbar, which is the RPU's move: every lane reads SOME lane's row, and what
// differs between its settings is only where that lane number comes from. REPLICATE
// IS LANE 0 BY CONVENTION: a value with no lane axis is the one a single bank holds.
// ARBITRARY READS ONE LANE NUMBER PER COLUMN THIS STAGE WALKS, and holds the caller
// to exactly that many -- a short queue used to fall back to the identity, which
// permuted the leading columns and left the rest in place without saying so.
crossLaneUnit_t::tile_t crossLaneUnit_t::rpu(uint32_t what, const tile_t &tile,
                                             const tile_t &pattern,
                                             const char *stage)
{
  if (what == XLU_RPU_BYPASS)
    return tile;
  //: A ROW IS AS WIDE AS THE WIDEST, AND WHAT IS MISSING IS ZERO -- not absent.
  //: A lane reading past the end of its source must still produce that column, or
  //: the next pop takes the following tile's first value and the answer is wrong
  //: rather than short. The width is the tile's own: `depth` on the way in, and the
  //: lane count once the crossing has made rows out of columns.
  size_t width = 0;
  for (uint32_t lane = 0; lane < n_lane; lane++)
    if (tile[lane].size() > width)
      width = tile[lane].size();
  //: AS MANY LANE NUMBERS AS THERE ARE COLUMNS, per lane, and the caller states
  //: them. The width is not the caller's to guess wrong about: a crossing turns
  //: `depth` columns into `n_lane` of them, and a queue sized for the other one
  //: used to be padded with the identity rather than refused.
  if (what == XLU_RPU_ARBITRARY) {
    for (uint32_t lane = 0; lane < n_lane; lane++)
      if (pattern[lane].size() != width) {
        fprintf(stderr, "XLU ERROR: the %s stage walks %zu columns but lane %u "
                        "carries %zu pattern entries (SIMM5 %u)\n",
                stage, width, lane, pattern[lane].size(), op);
        exit(INVALID_XLU_PATTERN);
      }
  }
  tile_t got(n_lane);
  for (uint32_t lane = 0; lane < n_lane; lane++) {
    for (size_t k = 0; k < width; k++) {
      uint32_t from = (what == XLU_RPU_ARBITRARY) ? pattern[lane][k] : 0u;
      bool have = from < n_lane && k < tile[from].size();
      got[lane].push_back(have ? tile[from][k] : 0u);
    }
  }
  return got;
}

// WHAT A POP DID NOT TAKE IS NOT THIS TILE'S, which `run` cleared before this; here
// each lane's row simply becomes its output queue.
void crossLaneUnit_t::emit(const tile_t &tile)
{
  for (uint32_t lane = 0; lane < n_lane && lane < tile.size(); lane++)
    for (size_t k = 0; k < tile[lane].size(); k++)
      out[lane]->push(tile[lane][k]);
}

