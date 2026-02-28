#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "yushkova_p_radix_sort_with_simple_merge/common/include/common.hpp"

namespace yushkova_p_radix_sort_with_simple_merge {

class YushkovaPRadixSortWithSimpleMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit YushkovaPRadixSortWithSimpleMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> sorted_data_;
};

}  // namespace yushkova_p_radix_sort_with_simple_merge
