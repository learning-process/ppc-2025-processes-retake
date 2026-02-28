#pragma once

#include "kaur_a_min_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kaur_a_min_matrix {

class KaurAMinMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KaurAMinMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace kaur_a_min_matrix
