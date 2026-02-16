#pragma once

#include "dergynov_s_hypercube/common/include/common.hpp"
#include "task/include/task.hpp"

namespace dergynov_s_hypercube {

class DergynovSHypercubeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit DergynovSHypercubeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace dergynov_s_hypercube