#pragma once

#include "klimov_m_lett_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimov_m_lett_count {

class KlimovMLettCountSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit KlimovMLettCountSEQ(const InputType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static int CountLettersInString(const char *data, int length);
};

}  // namespace klimov_m_lett_count