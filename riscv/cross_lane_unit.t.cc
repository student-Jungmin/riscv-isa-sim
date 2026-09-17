// What the cross-lane unit does, asked of the unit itself and not through a kernel.
// IT NEEDS NO PROCESSOR: the operations read `n_lane`, `depth` and the queues, so a
// unit built with a null processor is the whole machine. Build and run: ci-tests/test-xlu
#include "cross_lane_unit.h"

#include <cstdio>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

static int failures = 0;

static void check(bool ok, const char *what)
{
  if (!ok) {
    printf("FAILED: %s\n", what);
    failures++;
  }
}

// THE OPERATION IS ITS THREE FIELDS AND NOTHING ELSE. These are spelled out of the
// fields rather than named, because a name is a second place for a triple to live.
static uint32_t op(uint32_t pre, uint32_t xu, uint32_t post)
{
  return (pre << 3) | (xu << 2) | post;
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

// The same, with the PRE stage's pattern beside each value -- the `.ivv` push. One
// entry per value, so this queue is always exactly as long as `pre` is wide.
static void push_tile_p(crossLaneUnit_t &u,
                        const std::vector<std::vector<uint32_t> > &tile,
                        const std::vector<std::vector<uint32_t> > &pat)
{
  size_t depth = tile[0].size();
  for (size_t k = 0; k < depth; k++) {
    for (uint32_t lane = 0; lane < u.get_n_lane(); lane++)
      u.push_p(lane, lane < tile.size() ? tile[lane][k] : 0u,
               lane < pat.size() ? pat[lane][k] : lane);
    u.depth += 1;
  }
}

// The POST stage's pattern, loaded on its own: `want[lane]` repeated for as many
// columns as that stage will walk. It cannot ride with the data -- after a crossing
// the post stage walks `n_lane` columns while the tile is only `depth` deep.
static void load_post(crossLaneUnit_t &u, const std::vector<uint32_t> &want,
                      size_t width)
{
  for (size_t k = 0; k < width; k++)
    for (uint32_t lane = 0; lane < u.get_n_lane(); lane++)
      u.load_post(lane, lane < want.size() ? want[lane] : lane);
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
// and only `depth` lanes receive anything.
static void test_transpose()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(op(XLU_RPU_BYPASS, 1, XLU_RPU_BYPASS));
  push_tile(u, {{1, 2}, {3, 4}, {5, 6}, {7, 8}});
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 4);
  check(got[0] == std::vector<uint32_t>({1, 3, 5, 7}), "transpose: lane 0 takes column 0");
  check(got[1] == std::vector<uint32_t>({2, 4, 6, 8}), "transpose: lane 1 takes column 1");
  check(got[2].empty(), "transpose: a lane past the depth is given nothing");
}

static void test_broadcast()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(op(XLU_RPU_REPLICATE, 0, XLU_RPU_BYPASS));
  push_tile(u, {{9, 8}, {0, 0}, {0, 0}, {0, 0}});
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 2);
  for (uint32_t lane = 0; lane < 4; lane++)
    check(got[lane] == std::vector<uint32_t>({9, 8}), "broadcast: every lane gets lane 0's row");
}

// THE PRE PATTERN RIDES BESIDE THE DATA and costs the tile no row. It used to be row
// 0 of the tile, which cost a row and bound every row of the tile to one mapping.
static void test_permute()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(op(XLU_RPU_ARBITRARY, 0, XLU_RPU_BYPASS));
  push_tile_p(u, {{10}, {20}, {30}, {40}},
                 {{2},  {3},  {0},  {9}});   // 9 is past the end: that lane reads nothing
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 1);
  check(got[0] == std::vector<uint32_t>({30}), "permute: lane 0 reads lane 2");
  check(got[1] == std::vector<uint32_t>({40}), "permute: lane 1 reads lane 3");
  check(got[2] == std::vector<uint32_t>({10}), "permute: lane 2 reads lane 0");
  check(got[3] == std::vector<uint32_t>({0}),  "permute: a source past the end gives zero");
}

// A ROW MAY NAME ITS OWN SOURCE, which a pattern read out of row 0 never could: one
// row per column of the pre stage, so the two rows below take different mappings.
static void test_the_pre_pattern_may_differ_per_row()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(op(XLU_RPU_ARBITRARY, 0, XLU_RPU_BYPASS));
  push_tile_p(u, {{10, 11}, {20, 21}, {30, 31}, {40, 41}},
                 {{1,   2}, {1,   2}, {1,   2}, {1,   2}});
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 2);
  check(got[0] == std::vector<uint32_t>({20, 31}),
        "permute: column 0 reads lane 1 and column 1 reads lane 2");
}

// THE POST PATTERN IS LOADED, NOT PAIRED. After the crossing the stage walks
// `n_lane` columns while the tile is `depth` deep, so no push beside the data could
// be long enough -- which is why the load exists and why `.vvv` would not have helped.
static void test_post_pattern_after_a_crossing()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  load_post(u, {1, 0, 3, 2}, 4);        // the crossed tile is 4 columns wide
  u.set_op(op(XLU_RPU_BYPASS, 1, XLU_RPU_ARBITRARY));
  push_tile(u, {{1, 2}, {3, 4}, {5, 6}, {7, 8}});
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 4);
  // the crossing puts column 0 in lane 0 and column 1 in lane 1; post swaps them
  check(got[0] == std::vector<uint32_t>({2, 4, 6, 8}), "post: lane 0 reads lane 1's row");
  check(got[1] == std::vector<uint32_t>({1, 3, 5, 7}), "post: lane 1 reads lane 0's row");
}

// AS MANY LANE NUMBERS AS THERE ARE COLUMNS, and the caller states them. A short
// queue used to be padded with the identity, which permuted the leading columns and
// left the rest in place without saying so -- a wrong answer, not a trap.
static void test_a_short_pattern_traps()
{
  pid_t kid = fork();
  if (kid == 0) {
    crossLaneUnit_t u(0, 4);
    u.reset();
    load_post(u, {1, 0, 3, 2}, 2);      // two, where the stage walks four
    u.set_op(op(XLU_RPU_BYPASS, 1, XLU_RPU_ARBITRARY));
    push_tile(u, {{1, 2}, {3, 4}, {5, 6}, {7, 8}});
    u.run();
    _exit(0);                           // reached only if nothing refused
  }
  int status = 0;
  waitpid(kid, &status, 0);
  check(WIFEXITED(status) && WEXITSTATUS(status) == INVALID_XLU_PATTERN,
        "rpu: a pattern shorter than the stage is wide is refused");
}

// WHAT A POP DID NOT TAKE IS NOT THE NEXT TILE'S. A push covers every lane but only the
// used ones carry data, so an output column outlives the pop that read one register of
// it. Left in place it comes back as the next tile's first value -- a wrong answer and
// not a missing one.
static void test_no_residue_between_tiles()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(op(XLU_RPU_BYPASS, 1, XLU_RPU_BYPASS));
  push_tile(u, {{1, 2}, {3, 4}, {5, 6}, {7, 8}});
  u.run();
  (void)u.pop(0);                       // take ONE value and leave the rest standing
  u.set_op(op(XLU_RPU_BYPASS, 1, XLU_RPU_BYPASS));
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
  u.set_op(op(XLU_RPU_BYPASS, 1, XLU_RPU_BYPASS));
  push_tile(u, {{1}, {2}, {3}, {4}});
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 4);
  check(got[0] == std::vector<uint32_t>({1, 2, 3, 4}), "transpose: a depth-1 tile lands in lane 0");
  check(got[1].empty(), "transpose: and nowhere else");
}

// THE QUEUES CARRY 32 RAW BITS. Nothing here reads what they mean, so a float, a NaN
// and a lane number all cross as themselves.
static void test_bits_survive()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(op(XLU_RPU_BYPASS, 1, XLU_RPU_BYPASS));
  uint32_t bits[4] = {0x7fc00000u, 0xffffffffu, 0x80000000u, 0x00000001u};
  push_tile(u, {{bits[0]}, {bits[1]}, {bits[2]}, {bits[3]}});
  u.run();
  for (int i = 0; i < 4; i++)
    check(u.pop(0) == bits[i], "transpose: the bits come back as themselves");
}

// AN ALL-GATHER IS ONE PASS, and it is one because the post stage exists: cross, then
// hand lane 0's row -- which is now a depth slice -- back to every lane.
static void test_all_gather()
{
  crossLaneUnit_t u(0, 4);
  u.reset();
  u.set_op(op(XLU_RPU_BYPASS, 1, XLU_RPU_REPLICATE));
  push_tile(u, {{10}, {20}, {30}, {40}});
  u.run();
  std::vector<std::vector<uint32_t> > got = drain(u, 4);
  for (uint32_t lane = 0; lane < 4; lane++)
    check(got[lane] == std::vector<uint32_t>({10, 20, 30, 40}),
          "all-gather: every lane ends with every lane's value");
}

// PRE = 3 IS NOT A PASS. The fourth pre value names no RPU mode, which is what makes
// the eight codes it heads free to mean "load" instead of "run".
static void test_the_fields_decompose()
{
  check(xlu_pre(op(2, 1, 1)) == XLU_RPU_ARBITRARY && xlu_xu(op(2, 1, 1)) == 1
        && xlu_post(op(2, 1, 1)) == XLU_RPU_REPLICATE, "SIMM5 21 is arbitrary/swap/replicate");
  check(!xlu_is_load(op(2, 1, 1)), "a pass is not a load");
  check(xlu_is_load(XLU_LOAD_POST_PATTERN), "the post-pattern load is not a pass");
  check(xlu_pre(XLU_LOAD_POST_PATTERN) == XLU_RPU_NONE, "and it is pre = 3 that says so");
}

int main()
{
  test_all_gather();
  test_the_fields_decompose();
  test_transpose();
  test_broadcast();
  test_permute();
  test_the_pre_pattern_may_differ_per_row();
  test_post_pattern_after_a_crossing();
  test_a_short_pattern_traps();
  test_no_residue_between_tiles();
  test_depth_is_counted_not_assumed();
  test_bits_survive();
  if (failures)
    printf("%d FAILED\n", failures);
  else
    printf("cross_lane_unit: all checks passed\n");
  return failures ? 1 : 0;
}
