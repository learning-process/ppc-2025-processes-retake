#pragma once

#include "cheremkhin_a_gaus_vert/common/include/common.hpp"
#include "task/include/task.hpp"

namespace cheremkhin_a_gaus_vert {

class CheremkhinAGausVertMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit CheremkhinAGausVertMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace cheremkhin_a_gaus_vert
