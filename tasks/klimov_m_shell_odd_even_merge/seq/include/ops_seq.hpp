#pragma once

#include "klimov_m_shell_odd_even_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimov_m_shell_odd_even_merge {

class ShellBatcherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ShellBatcherSEQ(const InputType &input);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace klimov_m_shell_odd_even_merge
