#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "util/include/util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnikovEMaxValInColOfMatSEQ::LuchnikovEMaxValInColOfMatSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  matrix_ = in;
  result_.clear();
}

bool LuchnikovEMaxValInColOfMatSEQ::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  size_t cols = matrix[0].size();
  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool LuchnikovEMaxValInColOfMatSEQ::PreProcessingImpl() {
  const auto &matrix = GetInput();

  if (!matrix.empty()) {
    size_t cols = matrix[0].size();
    result_.assign(cols, std::numeric_limits<int>::min());
  }

  return true;
}

bool LuchnikovEMaxValInColOfMatSEQ::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  size_t rows = matrix.size();
  size_t cols = matrix[0].size();

  for (size_t j = 0; j < cols; ++j) {
    int max_val = std::numeric_limits<int>::min();
    for (size_t i = 0; i < rows; ++i) {
      if (matrix[i][j] > max_val) {
        max_val = matrix[i][j];
      }
    }
    result_[j] = max_val;
  }

  return true;
}

bool LuchnikovEMaxValInColOfMatSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return !result_.empty();
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
