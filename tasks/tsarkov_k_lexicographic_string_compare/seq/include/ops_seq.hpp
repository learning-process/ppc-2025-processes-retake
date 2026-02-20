#pragma once

#include "task/include/task.hpp"
#include "tsarkov_k_lexicographic_string_compare/common/include/common.hpp"

namespace tsarkov_k_lexicographic_string_compare {

class TsarkovKLexicographicStringCompareSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TsarkovKLexicographicStringCompareSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace tsarkov_k_lexicographic_string_compare
