// What the cross-lane unit does, asked of the unit itself and not through a kernel.
// IT NEEDS NO PROCESSOR: the operations read `n_lane`, `depth` and the queues, so a
// unit built with a null processor is the whole machine. Build and run: ci-tests/test-xlu
#include "cross_lane_unit.h"

#include <cstdio>
#include <vector>

static int failures = 0;

static void check(bool ok, const char *what)
{
  if (!ok) {
    printf("FAILED: %s\n", what);
    failures++;
  }
}

// Push `tile[lane][k]` a column at a time, the way an instruction stream does: one
// push carries one value per lane, and `depth` counts the columns.
static void push_tile(crossLaneUnit_t &u, const std::vector<std::vector<uint32_t> > &tile)
{
  size_t depth = tile[0].size();
  for (size_t k = 0; k < depth; k++) {
    for (uint32_t lane = 0; lane < u.get_n_lane(); lane++)
      u.push(lane, lane < tile.size() ? tile[lane][k] : 0u);
    u.depth += 1;
  }
}

static std::vector<std::vector<uint32_t> > drain(crossLaneUnit_t &u, size_t per_lane)
{
  std::vector<std::vector<uint32_t> > out(u.get_n_lane());
  for (uint32_t lane = 0; lane < u.get_n_lane(); lane++)
    for (size_t i = 0; i < per_lane && !u.out_empty(lane); i++)
      out[lane].push_back(u.pop(lane));
  return out;
}

// A TRANSPOSE MAKES THE DEPTH INDEX THE LANE INDEX. Column k becomes lane k's row,
// and only `depth` lanes receive anything -- the tile was n_lane wide and depth deep,
// so transposed it is depth wide.
static void test_transpose()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(XLU_TRANSPOSE);
  push_tile(u, {{1, 2}, {3, 4}, {5, 6}, {7, 8}});
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 4);
  check(got[0] == std::vector<uint32_t>({1, 3, 5, 7}), "transpose: lane 0 takes column 0");
  check(got[1] == std::vector<uint32_t>({2, 4, 6, 8}), "transpose: lane 1 takes column 1");
  check(got[2].empty(), "transpose: a lane past the depth is given nothing");
}

// ONE BANK'S ROW TO EVERY LANE, and lane 0 is the bank by convention: a value with no
// lane axis is the one a single bank holds, and a DMA that staged one element staged
// it there.
static void test_broadcast()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(XLU_BROADCAST);
  push_tile(u, {{9, 8}, {0, 0}, {0, 0}, {0, 0}});
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 2);
  for (uint32_t lane = 0; lane < 4; lane++)
    check(got[lane] == std::vector<uint32_t>({9, 8}), "broadcast: every lane gets lane 0's row");
}

// ROW 0 IS THE PATTERN AND NOT DATA -- one lane number per lane, saying where that lane
// READS FROM. It arrives down the same queue because an sf.vc form carries one vector
// operand and there is no second one to describe a mapping with.
static void test_permute()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(XLU_PERMUTE);
  //            pattern  row
  push_tile(u, {{2,      10},
                {3,      20},
                {0,      30},
                {9,      40}});      // 9 is past the end: that lane reads nothing
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 1);
  check(got[0] == std::vector<uint32_t>({30}), "permute: lane 0 reads lane 2");
  check(got[1] == std::vector<uint32_t>({40}), "permute: lane 1 reads lane 3");
  check(got[2] == std::vector<uint32_t>({10}), "permute: lane 2 reads lane 0");
  check(got[3] == std::vector<uint32_t>({0}),  "permute: a source past the end gives zero");
}

// WHAT A POP DID NOT TAKE IS NOT THE NEXT TILE'S. A push covers every lane but only the
// used ones carry data, so an output column outlives the pop that read one register of
// it. Left in place it comes back as the next tile's first value -- a wrong answer and
// not a missing one.
static void test_no_residue_between_tiles()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(XLU_TRANSPOSE);
  push_tile(u, {{1, 2}, {3, 4}, {5, 6}, {7, 8}});
  u.run();
  (void)u.pop(0);                       // take ONE value and leave the rest standing
  u.set_op(XLU_TRANSPOSE);
  push_tile(u, {{100, 0}, {200, 0}, {300, 0}, {400, 0}});
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 4);
  check(got[0] == std::vector<uint32_t>({100, 200, 300, 400}),
        "run: the output queue is emptied before the next tile");
}

// THE TILE IS NOT ASSUMED SQUARE. `depth` is counted from the pushes, so a 4x1 tile
// transposes into one lane holding four values.
static void test_depth_is_counted_not_assumed()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(XLU_TRANSPOSE);
  push_tile(u, {{1}, {2}, {3}, {4}});
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 4);
  check(got[0] == std::vector<uint32_t>({1, 2, 3, 4}), "transpose: a depth-1 tile lands in lane 0");
  check(got[1].empty(), "transpose: and nowhere else");
}

// EVERY BIT PATTERN CROSSES UNCHANGED. The queues carry 32 RAW BITS and nothing in
// the unit reads what they mean, so a signalling NaN, a denormal and a set sign bit
// are all just words -- which is the property that lets an integer tile cross at all.
static void test_bits_survive()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(XLU_TRANSPOSE);
  const uint32_t bits[4] = {0xFFFFFFFFu, 0x7F800001u, 0x00000001u, 0x80000000u};
  std::vector<std::vector<uint32_t> > tile(4);
  for (int i = 0; i < 4; i++)
    tile[i].push_back(bits[i]);
  push_tile(u, tile);
  u.run();
  for (int i = 0; i < 4; i++)
    check(u.pop(0) == bits[i], "transpose: the bits come back as themselves");
}

// ONE PASS, AND THAT IS THE WHOLE POINT OF THE FIELDS. An all-gather is the crossing
// followed by a replicate: after the crossing lane 0 holds every lane's value, and the
// post stage sends that row back to all of them. It used to be two instructions with a
// vector register between them, because a flat operation code had no seat for "and
// then shuffle again".
static void test_all_gather()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(XLU_ALL_GATHER);
  push_tile(u, {{10}, {20}, {30}, {40}});
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 4);
  for (uint32_t lane = 0; lane < 4; lane++)
    check(got[lane] == std::vector<uint32_t>({10, 20, 30, 40}),
          "all-gather: every lane ends with every lane's value");
}

// THE FIELDS ARE WHAT THE NAMES MEAN, so the names must decompose the way the encoding
// says. A name that drifted from its triple is an instruction spike and gem5 would
// disagree about, and the disagreement is a wrong answer rather than a failure.
static void test_the_names_are_their_fields()
{
  check(xlu_pre(XLU_TRANSPOSE) == XLU_RPU_BYPASS && xlu_xu(XLU_TRANSPOSE) == 1
        && xlu_post(XLU_TRANSPOSE) == XLU_RPU_BYPASS, "transpose is 0/1/0");
  check(xlu_pre(XLU_BROADCAST) == XLU_RPU_REPLICATE && xlu_xu(XLU_BROADCAST) == 0
        && xlu_post(XLU_BROADCAST) == XLU_RPU_BYPASS, "broadcast is replicate/0/0");
  check(xlu_pre(XLU_PERMUTE) == XLU_RPU_ARBITRARY && xlu_xu(XLU_PERMUTE) == 0
        && xlu_post(XLU_PERMUTE) == XLU_RPU_BYPASS, "permute is arbitrary/0/0");
  check(xlu_pre(XLU_ALL_GATHER) == XLU_RPU_BYPASS && xlu_xu(XLU_ALL_GATHER) == 1
        && xlu_post(XLU_ALL_GATHER) == XLU_RPU_REPLICATE, "all-gather is 0/1/replicate");
}

int main()
{
  test_all_gather();
  test_the_names_are_their_fields();
  test_transpose();
  test_broadcast();
  test_permute();
  test_no_residue_between_tiles();
  test_depth_is_counted_not_assumed();
  test_bits_survive();
  if (failures)
    printf("%d FAILED\n", failures);
  else
    printf("cross_lane_unit: all checks passed\n");
  return failures ? 1 : 0;
}
