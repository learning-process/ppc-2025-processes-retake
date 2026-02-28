#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <algorithm>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnilkovEMaxValInColOfMatSEQ::LuchnilkovEMaxValInColOfMatSEQ(const InType &in) : matrix_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool LuchnilkovEMaxValInColOfMatSEQ::ValidationImpl() {
  return !matrix_.empty() && !matrix_[0].empty();
}

bool LuchnilkovEMaxValInColOfMatSEQ::PreProcessingImpl() {
  return true;
}

bool LuchnilkovEMaxValInColOfMatSEQ::RunImpl() {
  size_t cols = matrix_[0].size();
  result_.resize(cols, std::numeric_limits<int>::min());

  for (const auto &row : matrix_) {
    for (size_t j = 0; j < cols; ++j) {
      result_[j] = std::max(result_[j], row[j]);
    }
  }
  return true;
}

bool LuchnilkovEMaxValInColOfMatSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
