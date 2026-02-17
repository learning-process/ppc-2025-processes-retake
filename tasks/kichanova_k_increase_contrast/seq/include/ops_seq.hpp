#pragma once

#include "kichanova_k_increase_contrast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kichanova_k_increase_contrast {

class KichanovaKIncreaseContrastSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KichanovaKIncreaseContrastSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kichanova_k_increase_contrast