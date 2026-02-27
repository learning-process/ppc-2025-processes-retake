#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace nazarova_k_gaussian_vert_scheme_processes {

// Augmented matrix for a linear system Ax=b:
// augmented is row-major with shape n x (n+1): [A | b]
struct Input {
  int n = 0;
  std::vector<double> augmented;
};

using InType = Input;
using OutType = std::vector<double>;            // solution vector x (size n)
using TestType = std::tuple<int, std::string>;  // (n, label) for gtest naming
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nazarova_k_gaussian_vert_scheme_processes
