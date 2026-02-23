#pragma once

#include "task/include/task.hpp"
#include "safaryan_a_sum_matrix  /common/include/common.hpp"

namespace safaryan_a_sum_matrix   {

class SafaryanASumMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SafaryanASumMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace safaryan_a_sum_matrix