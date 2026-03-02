#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnilkovEMaxValInColOfMatSeq::LuchnilkovEMaxValInColOfMatSeq(const InType &in) : matrix_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  result_.clear();
}

bool LuchnilkovEMaxValInColOfMatSeq::ValidationImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  const size_t cols = matrix[0].size();
  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool LuchnilkovEMaxValInColOfMatSeq::PreProcessingImpl() {
  const auto &matrix = GetInput();

  if (!matrix.empty()) {
    const size_t cols = matrix[0].size();
    result_.assign(cols, std::numeric_limits<int>::min());
  }

  return true;
}

bool LuchnilkovEMaxValInColOfMatSeq::RunImpl() {
  const auto &matrix = GetInput();

  if (matrix.empty()) {
    return false;
  }

  const size_t row_count = matrix.size();
  const size_t col_count = matrix[0].size();

  for (size_t j = 0; j < col_count; ++j) {
    int max_val = std::numeric_limits<int>::min();
    for (size_t i = 0; i < row_count; ++i) {
      max_val = std::max(max_val, matrix[i][j]);
    }
    result_[j] = max_val;
  }

  return true;
}

bool LuchnilkovEMaxValInColOfMatSeq::PostProcessingImpl() {
  GetOutput() = result_;
  return !result_.empty();
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
