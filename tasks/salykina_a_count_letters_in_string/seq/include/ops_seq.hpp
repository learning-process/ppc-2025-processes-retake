#pragma once
#include "salykina_a_count_letters_in_string/common/include/common.hpp"
#include "task/include/task.hpp"

namespace salykina_a_count_letters_in_string {

class SalykinaACountLettersSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SalykinaACountLettersSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace salykina_a_count_letters_in_string
