#pragma once

#include <vector>

#include "dergynov_s_radix_sort_double_simple_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dergynov_s_radix_sort_double_simple_merge {

class DergynovSRadixSortDoubleSimpleMergeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit DergynovSRadixSortDoubleSimpleMergeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> sorted_;
};

}  // namespace dergynov_s_radix_sort_double_simple_merge
