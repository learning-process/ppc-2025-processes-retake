#pragma once

#include "fedoseev_count_words_in_string/common/include/common.hpp"
#include "task/include/task.hpp"

namespace fedoseev_count_words_in_string {

class FedoseevCountWordsInStringSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit FedoseevCountWordsInStringSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace fedoseev_count_words_in_string
