#include "safaryan_a_sum_matrix/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "safaryan_a_sum_matrix/common/include/common.hpp"

namespace safaryan_a_sum_matrix {

SafaryanASumMatrixSEQ::SafaryanASumMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool SafaryanASumMatrixSEQ::ValidationImpl() {
  int rows = std::get<1>(GetInput());
  int cols = std::get<2>(GetInput());
  const auto &matrix_data = std::get<0>(GetInput());

  return matrix_data.size() == static_cast<size_t>(rows) * static_cast<size_t>(cols) && rows > 0 && cols > 0;
}

bool SafaryanASumMatrixSEQ::PreProcessingImpl() {
  int rows = std::get<1>(GetInput());
  GetOutput().resize(rows, 0);
  return true;
}

bool SafaryanASumMatrixSEQ::RunImpl() {
  int rows = std::get<1>(GetInput());
  int cols = std::get<2>(GetInput());
  const auto &matrix_data = std::get<0>(GetInput());

  for (int i = 0; i < rows; ++i) {
    int row_sum = 0;
    for (int j = 0; j < cols; ++j) {
      row_sum += matrix_data[(i * cols) + j];
    }
    GetOutput()[i] = row_sum;
  }
  return true;
}

bool SafaryanASumMatrixSEQ::PostProcessingImpl() {
  return true;
}
}  // namespace safaryan_a_sum_matrix
