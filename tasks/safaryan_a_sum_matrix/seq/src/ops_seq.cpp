#include "safaryan_a_sum_matrix/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "safaryan_a_sum_matrix/common/include/common.hpp"

namespace safaryan_a_sum_matrix {

SafaryanASumMatrixSEQ::SafaryanASumMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().assign(in.begin(), in.end());
}

bool SafaryanASumMatrixSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool SafaryanASumMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().resize(GetInput().size());
  std::fill(GetOutput().begin(), GetOutput().end(), 0);
  return true;
}

bool SafaryanASumMatrixSEQ::RunImpl() {
  for (size_t i = 0; i < GetInput().size(); ++i) {
    int row_sum = 0;
    for (int val : GetInput()[i]) {
      row_sum += val;
    }
    GetOutput()[i] = row_sum;
  }
  return true;
}

bool SafaryanASumMatrixSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}
}  // namespace safaryan_a_sum_matrix
