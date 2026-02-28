#include "salena_s_matrix_vector_mult/seq/include/ops_seq.hpp"

namespace salena_s_matrix_vector_mult {

TestTaskSEQ::TestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  if (GetInput().rows > 0) {
    GetOutput().resize(GetInput().rows, 0.0);
  }
}

bool TestTaskSEQ::ValidationImpl() {
  if (GetInput().rows <= 0 || GetInput().cols <= 0) return false;
  if (GetInput().matrix.size() != static_cast<size_t>(GetInput().rows * GetInput().cols)) return false;
  if (GetInput().vec.size() != static_cast<size_t>(GetInput().cols)) return false;
  return true;
}

bool TestTaskSEQ::PreProcessingImpl() {
  GetOutput().assign(GetInput().rows, 0.0);
  return true;
}

bool TestTaskSEQ::RunImpl() {
  const auto& matrix = GetInput().matrix;
  const auto& vec = GetInput().vec;
  int rows = GetInput().rows;
  int cols = GetInput().cols;
  auto& result = GetOutput();

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result[i] += matrix[i * cols + j] * vec[j];
    }
  }
  return true;
}

bool TestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace salena_s_matrix_vector_mult