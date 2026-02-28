#pragma once
#include "task/include/task.hpp"
#include <vector>

namespace salena_s_matrix_vector_mult {

struct MatVecIn {
  std::vector<double> matrix;
  std::vector<double> vec;
  int rows;
  int cols;
};

using InType = MatVecIn;
using OutType = std::vector<double>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace salena_s_matrix_vector_mult