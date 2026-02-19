#pragma once

#include <vector>

#include "task/include/task.hpp"
#include "vinyaikina_e_matrix_sum/common/include/common.hpp"

namespace vinyaikina_e_matrix_sum {

class VinyaikinaEMatrixSumSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VinyaikinaEMatrixSumSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rows_{0};
  int cols_{0};
  std::vector<int> matrix_;
  std::vector<int> row_sums_;
};

}  // namespace vinyaikina_e_matrix_sum
