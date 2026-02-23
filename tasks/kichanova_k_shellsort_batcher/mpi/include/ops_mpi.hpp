#pragma once

#include <vector>

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

  std::vector<int> GenerateLocalData(InType n, int rank, int size);
  void PerformOddEvenSort(std::vector<int> &local_data, int rank, int size);
  int GetPartner(int phase, int rank) const;
  std::int64_t CalculateChecksum(const std::vector<int> &data) const;

  static void ShellSort(std::vector<int> &arr);
  static void ExchangeAndMerge(std::vector<int> &local_data, int partner, int rank, int tag);
};

}  // namespace kichanova_k_shellsort_batcher
