#pragma once

#include "task/include/task.hpp"
#include "tsarkov_k_monte_carlo_integration/common/include/common.hpp"

namespace tsarkov_k_monte_carlo_integration {

class TsarkovKMonteCarloIntegrationMPI final : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit TsarkovKMonteCarloIntegrationMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace tsarkov_k_monte_carlo_integration
