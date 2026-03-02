#pragma once

#include "klimov_m_torus/common/include/common.hpp"
#include "task/include/task.hpp"

namespace klimov_m_torus {

class TorusReferenceImpl : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TorusReferenceImpl(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  InType local_request_{};
  OutType local_response_{};
};

}  // namespace klimov_m_torus
