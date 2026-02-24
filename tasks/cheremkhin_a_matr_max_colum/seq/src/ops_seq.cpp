#include "cheremkhin_a_matr_max_colum/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cheremkhin_a_matr_max_colum/common/include/common.hpp"

namespace cheremkhin_a_matr_max_colum {

CheremkhinAMatrMaxColumSEQ::CheremkhinAMatrMaxColumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().reserve(in.size());
  GetInput() = in;
  GetOutput() = {};
}

bool CheremkhinAMatrMaxColumSEQ::ValidationImpl() {
  return (!GetInput().empty());
}

bool CheremkhinAMatrMaxColumSEQ::PreProcessingImpl() {
  return true;
}

bool CheremkhinAMatrMaxColumSEQ::RunImpl() {
  const std::vector<std::vector<int>> &matrix = GetInput();
  std::vector<int> max_value_in_colum(matrix[0].size());

  for (std::size_t i = 0; i < matrix[0].size(); ++i) {
    int max_in_col = matrix[0][i];
    for (std::size_t j = 1; j < matrix.size(); ++j) {
      max_in_col = std::max(max_in_col, matrix[j][i]);
    }
    max_value_in_colum[i] = max_in_col;
  }

  GetOutput() = max_value_in_colum;
  return true;
}

bool CheremkhinAMatrMaxColumSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace cheremkhin_a_matr_max_colum
