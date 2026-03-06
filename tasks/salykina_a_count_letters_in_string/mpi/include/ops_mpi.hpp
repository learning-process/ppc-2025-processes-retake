#pragma once
#include "salykina_a_count_letters_in_string/common/include/common.hpp"
#include "task/include/task.hpp"

namespace salykina_a_count_letters_in_string {

class SalykinaACountLettersMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SalykinaACountLettersMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace salykina_a_count_letters_in_string
