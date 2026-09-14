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
enum xlu_op_t {
  XLU_TRANSPOSE  = 0,      // SIMM5 0
  XLU_BROADCAST  = 0x12,   // SIMM5 2
  XLU_PERMUTE    = 0x13,   // SIMM5 3
};

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
  void transpose(const std::vector<std::vector<uint32_t> > &tile);
  void broadcast(const std::vector<std::vector<uint32_t> > &tile);
  void permute(const std::vector<std::vector<uint32_t> > &tile);

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
