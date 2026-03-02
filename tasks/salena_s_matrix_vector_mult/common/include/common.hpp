#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace salena_s_matrix_vector_mult {

struct InType {
  int rows = 0;
  int cols = 0;
  std::vector<double> matrix;
  std::vector<double> vec;
};

using OutType = std::vector<double>;
using TestType = std::tuple<int, int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace salena_s_matrix_vector_mult
