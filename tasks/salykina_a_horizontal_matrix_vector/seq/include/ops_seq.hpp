#pragma once

#include "salykina_a_horizontal_matrix_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace salykina_a_horizontal_matrix_vector {

class SalykinaAHorizontalMatrixVectorSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit SalykinaAHorizontalMatrixVectorSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace salykina_a_horizontal_matrix_vector
