#pragma once

#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace yushkova_p_radix_sort_with_simple_merge {

using InType = std::vector<double>;
using OutType = std::tuple<std::vector<double>, int>;
using BaseTask = ppc::task::Task<InType, OutType>;

inline std::uint64_t EncodeDoubleKey(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(double));

  if ((bits & (1ULL << 63)) != 0ULL) {
    return ~bits;
  }

  return bits | (1ULL << 63);
}

inline double DecodeDoubleKey(std::uint64_t key) {
  std::uint64_t bits = 0;
  if ((key & (1ULL << 63)) != 0ULL) {
    bits = key & ~(1ULL << 63);
  } else {
    bits = ~key;
  }

  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(double));
  return value;
}

}  // namespace yushkova_p_radix_sort_with_simple_merge
