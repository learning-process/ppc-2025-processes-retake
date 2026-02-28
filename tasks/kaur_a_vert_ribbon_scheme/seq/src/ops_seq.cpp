#include "kaur_a_vert_ribbon_scheme/seq/include/ops_seq.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "kaur_a_vert_ribbon_scheme/common/include/common.hpp"

namespace kaur_a_vert_ribbon_scheme {

KaurAVertRibbonSchemeSEQ::KaurAVertRibbonSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool KaurAVertRibbonSchemeSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.rows <= 0 || input.cols <= 0) {
    return false;
  }
  if (input.matrix.size() != static_cast<std::size_t>(input.rows) * input.cols) {
    return false;
  }
  if (input.vector.size() != static_cast<std::size_t>(input.cols)) {
    return false;
  }
  return true;
}

bool KaurAVertRibbonSchemeSEQ::PreProcessingImpl() {
  const auto &input = GetInput();
  rows_ = input.rows;
  cols_ = input.cols;
  matrix_ = input.matrix;
  vector_ = input.vector;
  result_.assign(rows_, 0.0);
  return true;
}

bool KaurAVertRibbonSchemeSEQ::RunImpl() {
  for (int j = 0; j < cols_; j++) {
    for (int i = 0; i < rows_; i++) {
      result_[i] += matrix_[static_cast<std::size_t>(j * rows_) + i] * vector_[j];
    }
  }
  return true;
}

bool KaurAVertRibbonSchemeSEQ::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace kaur_a_vert_ribbon_scheme
