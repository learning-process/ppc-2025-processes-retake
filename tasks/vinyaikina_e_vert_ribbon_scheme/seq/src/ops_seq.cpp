#include "vinyaikina_e_vert_ribbon_scheme/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "vinyaikina_e_vert_ribbon_scheme/common/include/common.hpp"

namespace vinyaikina_e_vert_ribbon_scheme {

VinyaikinaEVertRibbonSchemeSEQ::VinyaikinaEVertRibbonSchemeSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool VinyaikinaEVertRibbonSchemeSEQ::ValidationImpl() {
  return GetInput() > 0 && GetOutput() == 0;
}

bool VinyaikinaEVertRibbonSchemeSEQ::PreProcessingImpl() {
  rows_ = GetInput();
  cols_ = GetInput();
  if (rows_ <= 0 || cols_ <= 0) {
    return false;
  }
  matrix_.assign(static_cast<std::size_t>(rows_) * cols_, 1);
  vector_.assign(cols_, 1);
  result_.assign(rows_, 0);
  return true;
}

bool VinyaikinaEVertRibbonSchemeSEQ::RunImpl() {
  for (int i = 0; i < rows_; i++) {
    int sum = 0;
    for (int j = 0; j < cols_; j++) {
      sum += matrix_[(i * cols_) + j] * vector_[j];
    }
    result_[i] = sum;
  }
  GetOutput() = 0;
  for (int i = 0; i < rows_; i++) {
    GetOutput() += result_[i];
  }
  return true;
}

bool VinyaikinaEVertRibbonSchemeSEQ::PostProcessingImpl() {
  if (GetInput() == 0) {
    return false;
  }
  GetOutput() /= GetInput();
  return true;
}

}  // namespace vinyaikina_e_vert_ribbon_scheme
