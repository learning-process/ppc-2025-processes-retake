#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnilkovEMaxValInColOfMatSEQ::LuchnilkovEMaxValInColOfMatSEQ(const InType &in) : matrix_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  result_.clear();
}

bool LuchnilkovEMaxValInColOfMatSEQ::ValidationImpl() {
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

bool LuchnilkovEMaxValInColOfMatSEQ::PreProcessingImpl() {
  const auto &matrix = GetInput();

  if (!matrix.empty()) {
    size_t cols = matrix[0].size();
    result_.assign(cols, std::numeric_limits<int>::min());
  }

  return true;
}

bool LuchnilkovEMaxValInColOfMatSEQ::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  size_t row_count = matrix.size();
  size_t col_count = matrix[0].size();

  std::vector<std::vector<int>> transposed(col_count, std::vector<int>(row_count));

  for (size_t i = 0; i < row_count; ++i) {
    for (size_t j = 0; j < col_count; ++j) {
      transposed[j][i] = matrix[i][j];
    }
  }

  for (size_t j = 0; j < col_count; ++j) {
    auto max_iter = std::max_element(transposed[j].begin(), transposed[j].end());
    result_[j] = *max_iter;
  }

  return true;
}

bool LuchnilkovEMaxValInColOfMatSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return !result_.empty();
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
