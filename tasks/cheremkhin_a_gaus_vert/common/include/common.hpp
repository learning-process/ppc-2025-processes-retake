#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace cheremkhin_a_gaus_vert {

struct Input {
  int n{};
  std::vector<double> a;  // row-major, size n*n
  std::vector<double> b;  // size n
};

using InType = Input;
using OutType = std::vector<double>;
using TestType = std::tuple<InType, OutType>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace cheremkhin_a_gaus_vert
