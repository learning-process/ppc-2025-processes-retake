#include "vinyaikina_e_matrix_sum/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "vinyaikina_e_matrix_sum/common/include/common.hpp"

namespace vinyaikina_e_matrix_sum {

VinyaikinaEMatrixSumSEQ::VinyaikinaEMatrixSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VinyaikinaEMatrixSumSEQ::ValidationImpl() {
  auto &input = GetInput();
  if (input.size() < 2) {
    return false;
  }
  int r = input[0];
  int c = input[1];
  if (r <= 0 || c <= 0) {
    return false;
  }
  return input.size() == (static_cast<size_t>(r) * static_cast<size_t>(c)) + 2;
}

bool VinyaikinaEMatrixSumSEQ::PreProcessingImpl() {
  auto &input = GetInput();
  rows_ = input[0];
  cols_ = input[1];
  matrix_.assign(input.begin() + 2, input.end());
  row_sums_.assign(rows_, 0);
  return true;
}

bool VinyaikinaEMatrixSumSEQ::RunImpl() {
  for (int i = 0; i < rows_; i++) {
    int sum = 0;
    for (int j = 0; j < cols_; j++) {
      sum += matrix_[(i * cols_) + j];
    }
    row_sums_[i] = sum;
  }
  return true;
}

bool VinyaikinaEMatrixSumSEQ::PostProcessingImpl() {
  GetOutput() = row_sums_;
  return true;
}

}  // namespace vinyaikina_e_matrix_sum
