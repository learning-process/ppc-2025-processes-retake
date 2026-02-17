#pragma once

#include "kichanova_k_shellsort_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kichanova_k_shellsort_batcher {

class KichanovaKShellsortBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit KichanovaKShellsortBatcherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ShellSort(std::vector<int> &arr);
  void ExchangeAndMerge(std::vector<int> &local_data, int partner, int rank, int tag);
};

}  // namespace kichanova_k_shellsort_batcher