#pragma once

#include "marov_radix_sort_double/common/include/common.hpp"
#include "task/include/task.hpp"

namespace marov_radix_sort_double {

class MarovRadixSortDoubleMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MarovRadixSortDoubleMPI(const InType& in);

 private:
  int proc_rank_{0};
  int proc_size_{1};

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace marov_radix_sort_double
