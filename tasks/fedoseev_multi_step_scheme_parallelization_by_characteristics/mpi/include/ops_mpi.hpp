#pragma once

#include "fedoseev_multi_step_scheme_parallelization_by_characteristics/common/include/common.hpp"
#include "task/include/task.hpp"

namespace fedoseev_multi_step_scheme_parallelization_by_characteristics {

class FedoseevMultiStepSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit FedoseevMultiStepSchemeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace fedoseev_multi_step_scheme_parallelization_by_characteristics
