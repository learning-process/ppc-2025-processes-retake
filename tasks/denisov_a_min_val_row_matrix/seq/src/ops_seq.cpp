#include "denisov_a_min_val_row_matrix/seq/include/ops_seq.hpp"

#include <algorithm>

#include "denisov_a_min_val_row_matrix/common/include/common.hpp"

namespace denisov_a_min_val_row_matrix {

DenisovAMinValRowMatrixSEQ::DenisovAMinValRowMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool DenisovAMinValRowMatrixSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return false;
  }

  if (!std::ranges::all_of(input, [](const auto &row) { return !row.empty(); })) {
    return false;
  }

  return true;
}

bool DenisovAMinValRowMatrixSEQ::PreProcessingImpl() {
  GetOutput().assign(GetInput().size(), 0);
  return true;
}

bool DenisovAMinValRowMatrixSEQ::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();
  int rows = static_cast<int>(input.size());

  for (int i = 0; i < rows; ++i) {
    output[i] = *std::min_element(input[i].begin(), input[i].end());
  }

  return true;
}

bool DenisovAMinValRowMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace denisov_a_min_val_row_matrix
