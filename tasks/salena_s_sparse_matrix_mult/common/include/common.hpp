#pragma once
#include <vector>
#include "task/include/task.hpp"

namespace salena_s_sparse_matrix_mult {

struct SparseMatrixCRS {
  int rows = 0;
  int cols = 0;
  std::vector<double> values;
  std::vector<int> col_indices;
  std::vector<int> row_ptr;
};

struct SparseMultIn {
  SparseMatrixCRS A;
  SparseMatrixCRS B;
};

using InType = SparseMultIn;
using OutType = SparseMatrixCRS;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace salena_s_sparse_matrix_mult