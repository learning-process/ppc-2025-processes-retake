#pragma once

#include <vector>

#include "sabutay_a_radix_sort_double_with_merge/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sabutay_a_radix_sort_double_with_merge {

class SabutayAradixSortDoubleWithMergeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit SabutayAradixSortDoubleWithMergeMPI(const InType &in) {
    SetTypeOfTask(GetStaticTypeOfTask());
    GetInput() = in;
    GetOutput() = {};
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> local_;
  std::vector<int> counts_;
  std::vector<int> displs_;
  int world_rank_{0};
  int world_size_{1};
};

}  // namespace sabutay_a_radix_sort_double_with_merge