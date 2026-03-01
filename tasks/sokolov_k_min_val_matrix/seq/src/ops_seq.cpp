#include "sokolov_k_min_val_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>

#include "sokolov_k_min_val_matrix/common/include/common.hpp"

namespace k_sokolov_min_val_matrix {

SokolovKMinValMatrixSEQ::SokolovKMinValMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SokolovKMinValMatrixSEQ::ValidationImpl() {
  return GetInput() > 0;
}

bool SokolovKMinValMatrixSEQ::PreProcessingImpl() {
  int n = GetInput();
  if (n <= 0) {
    rows_ = 0;
    cols_ = 0;
    return true;
  }
  rows_ = n;
  cols_ = n;
  matrix_.resize(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n * n; i++) {
    matrix_[i] = i + 1;
  }
  return true;
}

bool SokolovKMinValMatrixSEQ::RunImpl() {
  if (matrix_.empty()) {
    return true;
  }
  min_val_ = *std::ranges::min_element(matrix_);
  return true;
}

bool SokolovKMinValMatrixSEQ::PostProcessingImpl() {
  GetOutput() = min_val_;
  return true;
}

}  // namespace k_sokolov_min_val_matrix
