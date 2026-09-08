// The transpose unit: rows go in a vector register at a time, columns come back.
// ONE QUEUE PER LANE, like systolicArray_t -- a push IS the serialisation, and the
// transpose is the only thing that happens between the two queues.
#ifndef _RISCV_CROSS_LANE_UNIT_H
#define _RISCV_CROSS_LANE_UNIT_H

#include "processor.h"
#include <queue>
#include <vector>

class processor_t;

class crossLaneUnit_t
{
public:
  processor_t *p;
  uint32_t n_lane;
  // The tile on the way in and the tile on the way out. `in[L]` is lane L's row as
  // it was pushed; `out[L]` is lane L's column once `transpose` has run.
  std::queue<float> **in;
  std::queue<float> **out;
  // How deep each lane's row is -- the number of values pushed per lane since the
  // last transpose. IT IS THE TILE'S OTHER DIMENSION, so the transpose reads it
  // rather than assuming the tile is square.
  reg_t depth;

public:
  void reset();
  void transpose();

  crossLaneUnit_t(processor_t *p, reg_t n_vu) : p(p),
                                                n_lane(n_vu),
                                                in(0),
                                                out(0),
                                                depth(0)
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

  void push(uint32_t lane, float val)
  {
    in[lane]->push(val);
  }

  float pop(uint32_t lane)
  {
    float val = out[lane]->front();
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
