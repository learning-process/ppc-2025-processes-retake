#pragma once

#include "cheremkhin_a_radix_sort_batcher/common/include/common.hpp"
#include "task/include/task.hpp"

namespace cheremkhin_a_radix_sort_batcher {

class CheremkhinARadixSortBatcherMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit CheremkhinARadixSortBatcherMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace cheremkhin_a_radix_sort_batcher
