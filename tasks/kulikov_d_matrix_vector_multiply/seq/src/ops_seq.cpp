#include "kulikov_d_matrix_vector_multiply/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "kulikov_d_matrix_vector_multiply/common/include/common.hpp"

namespace kulikov_d_matrix_vector_multiply {

KulikovDMatrixMultiplySEQ::KulikovDMatrixMultiplySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KulikovDMatrixMultiplySEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.rows <= 0 || input.cols <= 0) {
    return false;
  }

  if (input.matrix.size() != static_cast<size_t>(input.rows) * static_cast<size_t>(input.cols)) {
    return false;
  }

  if (input.vector.size() != static_cast<size_t>(input.cols)) {
    return false;
  }

  return GetOutput().empty();
}

bool KulikovDMatrixMultiplySEQ::PreProcessingImpl() {
  const auto &input = GetInput();

  GetOutput().assign(input.rows, 0);

  return true;
}

bool KulikovDMatrixMultiplySEQ::RunImpl() {
  const auto &input = GetInput();
  auto &result = GetOutput();

  for (int i = 0; i < input.rows; i++) {
    int sum = 0;

    for (int j = 0; j < input.cols; j++) {
      sum += input.matrix[(i * input.cols) + j] * input.vector[j];
    }

    result[i] = sum;
  }

  return true;
}

bool KulikovDMatrixMultiplySEQ::PostProcessingImpl() {
  return true;
}

}  // namespace kulikov_d_matrix_vector_multiply
