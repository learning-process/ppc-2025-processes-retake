#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <vector>

namespace luchnikov_e_max_val_in_col_of_mat {

LuchnikovEMaxValInColOfMatSEQ::LuchnikovEMaxValInColOfMatSEQ(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<int>();
}

bool LuchnikovEMaxValInColOfMatSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return false;
  }
  rows_ = GetInput().size();
  cols_ = GetInput()[0].size();
  for (const auto& row : GetInput()) {
    if (row.size() != static_cast<size_t>(cols_)) {
      return false;
    }
  }
  return true;
}

bool LuchnikovEMaxValInColOfMatSEQ::PreProcessingImpl() {
  rows_ = GetInput().size();
  cols_ = GetInput()[0].size();
  return true;
}

bool LuchnikovEMaxValInColOfMatSEQ::RunImpl() {
  GetOutput().resize(cols_, INT_MIN);

  for (int j = 0; j < cols_; j++) {
    for (int i = 0; i < rows_; i++) {
      if (GetInput()[i][j] > GetOutput()[j]) {
        GetOutput()[j] = GetInput()[i][j];
      }
    }
  }

  return true;
}

bool LuchnikovEMaxValInColOfMatSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace luchnikov_e_max_val_in_col_of_mat