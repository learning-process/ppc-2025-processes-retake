#pragma once

#include <cstdint>
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

  [[nodiscard]] static std::vector<int> GenerateLocalData(InType n, int rank, int size);
  static void PerformOddEvenSort(std::vector<int> &local_data, int rank, int size);
  [[nodiscard]] static int GetPartner(int phase, int rank);
  [[nodiscard]] static std::int64_t CalculateChecksum(const std::vector<int> &data);

  static void ShellSort(std::vector<int> &arr);
  static void ExchangeAndMerge(std::vector<int> &local_data, int partner, int rank, int tag);
};

}  // namespace kichanova_k_shellsort_batcher
