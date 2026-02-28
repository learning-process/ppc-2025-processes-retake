#include "kaur_a_min_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kaur_a_min_matrix/common/include/common.hpp"

namespace kaur_a_min_matrix {

KaurAMinMatrixSEQ::KaurAMinMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KaurAMinMatrixSEQ::ValidationImpl() {
  auto &rows = std::get<0>(GetInput());
  auto &columns = std::get<1>(GetInput());
  auto &matrix = std::get<2>(GetInput());

  return (rows != 0 && columns != 0) && (rows * columns == matrix.size()) && (GetOutput() == 0);
}

bool KaurAMinMatrixSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool KaurAMinMatrixSEQ::RunImpl() {
  auto &matrix = std::get<2>(GetInput());

  if (matrix.empty()) {
    return false;
  }

  int min_val = matrix[0];
  for (size_t i = 1; i < matrix.size(); i++) {
    min_val = std::min(min_val, matrix[i]);
  }

  GetOutput() = min_val;
  return true;
}

bool KaurAMinMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kaur_a_min_matrix
