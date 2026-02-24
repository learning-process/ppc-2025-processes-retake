#pragma once

#include <vector>

#include "kichanova_k_shellsort_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kichanova_k_shellsort_batcher {

class KichanovaKShellsortBatcherSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit KichanovaKShellsortBatcherSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ShellSort(std::vector<int> &arr);
  static void OddEvenBatcherMerge(const std::vector<int> &left, const std::vector<int> &right,
                                  std::vector<int> &merged);
};

}  // namespace kichanova_k_shellsort_batcher
