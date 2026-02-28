#pragma once

#include "klimov_m_torus/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimov_m_torus {

class TorusSequential : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TorusSequential(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType localInput_{};
  OutType localOutput_{};
};

}  // namespace klimov_m_torus