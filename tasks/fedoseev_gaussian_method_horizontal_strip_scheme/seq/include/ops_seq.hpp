#pragma once

#include <cstddef>
#include <vector>

#include "fedoseev_gaussian_method_horizontal_strip_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace fedoseev_gaussian_method_horizontal_strip_scheme {

class FedoseevGaussianMethodHorizontalStripSchemeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit FedoseevGaussianMethodHorizontalStripSchemeSEQ(const InType &input_data);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static bool ForwardElimination(InType &matrix, size_t n, size_t cols);
  static size_t SelectPivotRow(const InType &matrix, size_t k, size_t n);
  static void EliminateRows(InType &matrix, size_t k, size_t n, size_t cols);
  static std::vector<double> BackwardSubstitution(const InType &matrix, size_t n, size_t cols);
};

}  // namespace fedoseev_gaussian_method_horizontal_strip_scheme