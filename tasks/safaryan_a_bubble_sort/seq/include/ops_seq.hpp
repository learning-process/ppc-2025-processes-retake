#pragma once

#include "safaryan_a_bubble_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace safaryan_a_bubble_sort {
class SafaryanABubbleSortSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SafaryanABubbleSortSEQ(const InType &in);

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace safaryan_a_bubble_sort