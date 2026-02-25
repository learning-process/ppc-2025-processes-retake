#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace salykina_a_sparse_matrix_mult {

struct SparseMatrixCRS {
  std::vector<double> values;
  std::vector<int> col_indices;
  std::vector<int> row_ptr;
  int num_rows{};
  int num_cols{};
};

struct MatrixMultiplicationInput {
  SparseMatrixCRS matrix_a;
  SparseMatrixCRS matrix_b;
};

using InType = MatrixMultiplicationInput;
using OutType = SparseMatrixCRS;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace salykina_a_sparse_matrix_mult
