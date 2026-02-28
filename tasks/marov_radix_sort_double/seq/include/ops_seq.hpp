#pragma once

#include "marov_radix_sort_double/common/include/common.hpp"
#include "task/include/task.hpp"

namespace marov_radix_sort_double {

class MarovRadixSortDoubleSeq : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MarovRadixSortDoubleSeq(const InType& in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace marov_radix_sort_double
