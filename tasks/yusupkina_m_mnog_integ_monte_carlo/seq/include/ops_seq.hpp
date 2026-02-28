#pragma once

#include "task/include/task.hpp"
#include "yusupkina_m_mnog_integ_monte_carlo/common/include/common.hpp"

namespace yusupkina_m_mnog_integ_monte_carlo {

class YusupkinaMMnogIntegMonteCarloSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit YusupkinaMMnogIntegMonteCarloSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace yusupkina_m_mnog_integ_monte_carlo
