#pragma once

#include "nazyrov_a_min_val_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nazyrov_a_min_val_vec {

class MinValVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MinValVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nazyrov_a_min_val_vec
