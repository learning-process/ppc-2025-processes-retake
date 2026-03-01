#pragma once

#include "denisov_a_ring/common/include/common.hpp"
#include "task/include/task.hpp"

namespace denisov_a_ring {

class RingTopologySEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RingTopologySEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace denisov_a_ring
