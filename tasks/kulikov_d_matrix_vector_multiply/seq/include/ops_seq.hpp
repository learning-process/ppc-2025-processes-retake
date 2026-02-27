#pragma once

#include "kulikov_d_matrix_vector_multiply/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kulikov_d_matrix_vector_multiply {

class KulikovDMatrixMultiplySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KulikovDMatrixMultiplySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kulikov_d_matrix_vector_multiply
