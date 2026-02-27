#include "zyuzin_n_multiplication_matrix_horiz/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "zyuzin_n_multiplication_matrix_horiz/common/include/common.hpp"

namespace zyuzin_n_multiplication_matrix_horiz {

ZyuzinNMultiplicationMatrixSEQ::ZyuzinNMultiplicationMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool ZyuzinNMultiplicationMatrixSEQ::ValidationImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &matrix_b = GetInput().second;

  if (matrix_a.empty() || matrix_b.empty()) {
    return false;
  }

  size_t cols_a = matrix_a[0].size();
  for (size_t i = 1; i < matrix_a.size(); i++) {
    if (matrix_a[i].size() != cols_a) {
      return false;
    }
  }

  size_t cols_b = matrix_b[0].size();
  for (size_t i = 1; i < matrix_b.size(); i++) {
    if (matrix_b[i].size() != cols_b) {
      return false;
    }
  }

  return cols_a == matrix_b.size();
}

bool ZyuzinNMultiplicationMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ZyuzinNMultiplicationMatrixSEQ::RunImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &matrix_b = GetInput().second;
  size_t rows_a = matrix_a.size();
  size_t cols_b = matrix_b[0].size();
  size_t cols_a = matrix_a[0].size();

  GetOutput().resize(rows_a);
  for (size_t i = 0; i < rows_a; ++i) {
    GetOutput()[i].assign(cols_b, 0.0);
  }

  for (size_t i = 0; i < rows_a; i++) {
    for (size_t j = 0; j < cols_b; j++) {
      for (size_t k = 0; k < cols_a; k++) {
        GetOutput()[i][j] += matrix_a[i][k] * matrix_b[k][j];
      }
    }
  }

  return true;
}

bool ZyuzinNMultiplicationMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace zyuzin_n_multiplication_matrix_horiz
