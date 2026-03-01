#pragma once

#include <vector>

#include "denisov_a_quick_sort_simple_merging/common/include/common.hpp"
#include "task/include/task.hpp"

namespace denisov_a_quick_sort_simple_merging {

class DenisovAQuickSortMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit DenisovAQuickSortMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void QuickSort(std::vector<int> &data, int begin, int end);
  static std::vector<int> Merge(const std::vector<int> &left_block, const std::vector<int> &right_block);
};

}  // namespace denisov_a_quick_sort_simple_merging
