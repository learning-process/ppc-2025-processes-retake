#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "yushkova_p_radix_sort_with_simple_merge/common/include/common.hpp"

namespace yushkova_p_radix_sort_with_simple_merge {

class YushkovaPRadixSortWithSimpleMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit YushkovaPRadixSortWithSimpleMergeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> merged_result_;
};

}  // namespace yushkova_p_radix_sort_with_simple_merge
