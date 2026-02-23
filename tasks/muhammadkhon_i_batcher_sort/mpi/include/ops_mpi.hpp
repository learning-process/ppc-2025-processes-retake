#pragma once

#include "muhammadkhon_i_batcher_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace muhammadkhon_i_batcher_sort {

class MuhammadkhonIBatcherSortMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit MuhammadkhonIBatcherSortMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace muhammadkhon_i_batcher_sort
