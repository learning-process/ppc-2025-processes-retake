#pragma once

#include "safaryan_a_sum_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace safaryan_a_sum_matrix {

class SafaryanASumMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit SafaryanASumMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace safaryan_a_sum_matrix
