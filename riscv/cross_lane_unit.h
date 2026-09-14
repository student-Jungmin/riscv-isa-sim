// The cross-lane unit: a tile goes in a vector register at a time and comes back
// with the LANE AXIS rewritten -- transposed, replicated or permuted. IT DOES NO
// ARITHMETIC: a reduction is a transpose and then the VECTOR UNIT's own fold, which
// is the half that knows the type. SO THE QUEUES CARRY 32 RAW BITS and nothing here
// reads what they mean -- an integer, a float, a NaN and a lane number all cross as
// the bits they are. ONE QUEUE PER
// LANE, like systolicArray_t; the push IS the serialisation and the op runs between.
#ifndef _RISCV_CROSS_LANE_UNIT_H
#define _RISCV_CROSS_LANE_UNIT_H

#include "processor.h"
#include <queue>
#include <vector>

class processor_t;

// WHAT THE PENDING TILE IS FOR. The push names it, the pop runs it: no lane's answer
// is known until every lane is in.
// The numbering is the ISA's SIMM5. ONE FAMILY: transpose, broadcast and permute
// are one machine -- this object, one queue pair, one `run()` -- and the encoding
// says so. The transpose held a family of its own while the crossbar COMBINED and
// it did not; with the reduce retired both only move, and SIMM5 0 is where the
// reduce-add that split them used to sit.
// THE FIVE BITS ARE THREE FIELDS, not a list of operations. What this unit can be
// asked is exactly three questions -- shuffle the lanes BEFORE crossing, change the
// depth WHILE crossing, shuffle them AFTER -- and an operation is a combination of
// answers rather than a name in a table. That is why an all-gather is ONE pass here
// and used to be two: a flat enumeration has no seat for "and then shuffle again",
// so the compiler had to pop the transpose out to a vector register and push it back.
//
//   SIMM5[4:3]  pre  -- RPU    0 bypass · 1 replicate (canned) · 2 arbitrary (pattern)
//   SIMM5[2]    XU             0 depth as it was · 1 depth <-> lane
//   SIMM5[1:0]  post -- RPU    the same three
//
// RPU APPEARS TWICE AND XU ONCE BECAUSE THAT IS THE HARDWARE: the RPU sits on both
// sides of the XU and the crossing always happens, which is why SIMM5 = 0 is a trap
// rather than a no-op -- there is no reason to enter the unit without crossing.
enum xlu_rpu_t {
  XLU_RPU_BYPASS    = 0,
  XLU_RPU_REPLICATE = 1,   // every lane reads lane 0
  XLU_RPU_ARBITRARY = 2,   // every lane reads the lane row 0 names for it
};

// The combinations the compiler asks for today. A name here is a shorthand for a
// field triple and never a fourth thing the unit knows how to do.
enum xlu_op_t {
  XLU_TRANSPOSE  = 4,      // pre bypass    · XU swap · post bypass
  XLU_BROADCAST  = 8,      // pre replicate · XU keep · post bypass
  XLU_PERMUTE    = 16,     // pre arbitrary · XU keep · post bypass
  XLU_ALL_GATHER = 5,      // pre bypass    · XU swap · post replicate
};

static inline uint32_t xlu_pre(xlu_op_t o)  { return ((uint32_t)o >> 3) & 3u; }
static inline uint32_t xlu_xu(xlu_op_t o)   { return ((uint32_t)o >> 2) & 1u; }
static inline uint32_t xlu_post(xlu_op_t o) { return (uint32_t)o & 3u; }

class crossLaneUnit_t
{
public:
  processor_t *p;
  uint32_t n_lane;
  // The tile on the way in and the tile on the way out. `in[L]` is lane L's row as
  // it was pushed; `out[L]` is lane L's column once `transpose` has run.
  std::queue<uint32_t> **in;
  std::queue<uint32_t> **out;
  // How deep each lane's row is -- the number of values pushed per lane since the
  // last transpose. IT IS THE TILE'S OTHER DIMENSION, so the transpose reads it
  // rather than assuming the tile is square.
  reg_t depth;
  // The op the pending tile was pushed for. A push sets it; two pushes with
  // different ops between one pop is the compiler's error, not a mode to model.
  xlu_op_t op;

public:
  void reset();
  void run();
  typedef std::vector<std::vector<uint32_t> > tile_t;
  tile_t crossing(const tile_t &tile);
  tile_t rpu(uint32_t what, const tile_t &tile);
  void emit(const tile_t &tile);


  crossLaneUnit_t(processor_t *p, reg_t n_vu) : p(p),
                                                n_lane(n_vu),
                                                in(0),
                                                out(0),
                                                depth(0),
                                                op(XLU_TRANSPOSE)
  {
  }

  ~crossLaneUnit_t()
  {
    free_queues();
  }

  void free_queues()
  {
    if (in) {
      for (uint32_t i = 0; i < n_lane; i++)
        delete in[i];
      delete[] in;
      in = 0;
    }
    if (out) {
      for (uint32_t i = 0; i < n_lane; i++)
        delete out[i];
      delete[] out;
      out = 0;
    }
  }

  void push(uint32_t lane, uint32_t val)
  {
    in[lane]->push(val);
  }

  void set_op(xlu_op_t o)
  {
    op = o;
  }

  uint32_t pop(uint32_t lane)
  {
    uint32_t val = out[lane]->front();
    out[lane]->pop();
    return val;
  }

  bool out_empty(uint32_t lane)
  {
    return out[lane]->empty();
  }

  // Whether a pop has anything to work with. A POP IS WHAT FIRES THE TRANSPOSE, so
  // this is what decides between "run it now" and "there is nothing here".
  bool has_input()
  {
    return depth > 0;
  }

  uint32_t get_n_lane()
  {
    return n_lane;
  }

  bool get_env_flag(const char* env)
  {
    const char* env_value = std::getenv(env);
    return env_value ? std::stoi(env_value) : 0;
  }
};

#endif // _RISCV_CROSS_LANE_UNIT_H
