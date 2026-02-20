#pragma once

#include "sabutay_vector_sign_changes/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sabutay_vector_sign_changes {

class SabutayVectorSignChangesSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SabutayVectorSignChangesSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sabutay_vector_sign_changes
