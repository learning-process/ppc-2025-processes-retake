#pragma once

#include "marov_count_letters/common/include/common.hpp"
#include "task/include/task.hpp"

namespace marov_count_letters {

class MarovCountLettersMpi : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MarovCountLettersMpi(const InType& in);

 private:
  int proc_rank_{0};
  int proc_size_{1};

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace marov_count_letters
