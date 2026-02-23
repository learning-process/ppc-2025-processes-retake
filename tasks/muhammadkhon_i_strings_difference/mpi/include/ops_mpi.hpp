#pragma once

#include "muhammadkhon_i_strings_difference/common/include/common.hpp"
#include "task/include/task.hpp"

namespace muhammadkhon_i_strings_difference {

class MuhammadkhonIStringsDifferenceMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MuhammadkhonIStringsDifferenceMPI(const InType &in);

 private:
  int proc_rank_{0};
  int proc_size_{1};

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace muhammadkhon_i_strings_difference