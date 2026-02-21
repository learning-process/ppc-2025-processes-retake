#pragma once

#include "tsarkov_k_hypercube/common/include/common.hpp"

namespace tsarkov_k_hypercube {

class TsarkovKHypercubeSEQ final : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit TsarkovKHypercubeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace tsarkov_k_hypercube
