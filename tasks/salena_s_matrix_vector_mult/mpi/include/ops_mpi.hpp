#pragma once

#include <cstddef>
#include <vector>

#include "salena_s_matrix_vector_mult/common/include/common.hpp"

namespace salena_s_matrix_vector_mult {

class TestTaskMPI : public BaseTask {
 public:
  // NOLINTNEXTLINE(misc-include-cleaner)
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit TestTaskMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<double> Transpose(const std::vector<double> &matrix, int rows, int cols);
};

}  // namespace salena_s_matrix_vector_mult
