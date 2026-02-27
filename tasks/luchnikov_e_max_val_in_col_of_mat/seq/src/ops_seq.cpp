#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <vector>

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnikovEMaxValInColOfMatSEQ::LuchnikovEMaxValInColOfMatSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  input_copy_ = in;
  GetOutput() = std::vector<int>();
}

bool LuchnikovEMaxValInColOfMatSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }

  rows_ = static_cast<int>(GetInput().size());
  cols_ = static_cast<int>(GetInput()[0].size());

  for (const auto &row : GetInput()) {
    if (static_cast<int>(row.size()) != cols_) {
      return false;
    }
  }

  return GetOutput().empty();
}

bool LuchnikovEMaxValInColOfMatSEQ::PreProcessingImpl() {
  if (!GetInput().empty()) {
    cols_ = static_cast<int>(GetInput()[0].size());
    local_result_.assign(cols_, INT_MIN);
    input_copy_ = GetInput();
  }
  return true;
}

bool LuchnikovEMaxValInColOfMatSEQ::RunImpl() {
  if (input_copy_.empty()) {
    return false;
  }

  for (int j = 0; j < cols_; ++j) {
    int current_max = input_copy_[0][j];
    for (int i = 1; i < rows_; ++i) {
      int val = input_copy_[i][j];
      if (val > current_max) {
        current_max = val;
      }
    }
    local_result_[j] = current_max;
  }

  return true;
}

bool LuchnikovEMaxValInColOfMatSEQ::PostProcessingImpl() {
  GetOutput() = local_result_;
  return !local_result_.empty();
}

}  // namespace luchnikov_e_max_val_in_col_of_mat
