#pragma once

#include "task/include/task.hpp"
#include "yushkova_p_hypercube/common/include/common.hpp"

namespace yushkova_p_hypercube {

class YushkovaPHypercubeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit YushkovaPHypercubeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace yushkova_p_hypercube
