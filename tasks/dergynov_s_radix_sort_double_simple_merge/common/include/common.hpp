#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace dergynov_s_radix_sort_double_simple_merge {

using InType = std::vector<double>;
using OutType = std::tuple<std::vector<double>, int>;
using TestType = std::tuple<std::tuple<std::vector<double>, std::vector<double>>, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline uint64_t DoubleToSortableUint64(double d) {
  uint64_t u = 0;
  std::memcpy(&u, &d, sizeof(double));
  if ((u & 0x8000000000000000ULL) != 0U) {
    u = ~u;
  } else {
    u |= 0x8000000000000000ULL;
  }
  return u;
}

inline double SortableUint64ToDouble(uint64_t u) {
  if ((u & 0x8000000000000000ULL) != 0U) {
    u &= ~0x8000000000000000ULL;
  } else {
    u = ~u;
  }
  double d = 0.0;
  std::memcpy(&d, &u, sizeof(double));
  return d;
}

}  // namespace dergynov_s_radix_sort_double_simple_merge
