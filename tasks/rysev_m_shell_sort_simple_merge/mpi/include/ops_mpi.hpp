#pragma once

#include <vector>

#include "rysev_m_shell_sort_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rysev_m_shell_sort_simple_merge {

class RysevMShellSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit RysevMShellSortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void ShellSort(std::vector<int> &arr);

  bool DistributeData(const std::vector<int> &input_data, int data_size, std::vector<int> &send_counts,
                      std::vector<int> &displs, std::vector<int> &local_block) const;

  void MergeResults(int data_size, const std::vector<int> &send_counts, const std::vector<int> &displs,
                    const std::vector<int> &gathered_data);

  int rank_ = 0;
  int num_procs_ = 0;
  std::vector<int> local_block_;
};

}  // namespace rysev_m_shell_sort_simple_merge
