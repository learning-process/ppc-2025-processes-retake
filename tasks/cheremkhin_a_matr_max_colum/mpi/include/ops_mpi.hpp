#pragma once

#include "cheremkhin_a_matr_max_colum/common/include/common.hpp"
#include "task/include/task.hpp"

namespace cheremkhin_a_matr_max_colum {

class CheremkhinAMatrMaxColumMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit CheremkhinAMatrMaxColumMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace cheremkhin_a_matr_max_colum
