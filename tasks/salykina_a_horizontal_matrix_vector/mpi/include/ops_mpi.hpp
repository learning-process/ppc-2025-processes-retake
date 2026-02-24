#pragma once

#include <vector>

#include "salykina_a_horizontal_matrix_vector/common/include/common.hpp"
#include "task/include/task.hpp"

namespace salykina_a_horizontal_matrix_vector {

class SalykinaAHorizontalMatrixVectorMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SalykinaAHorizontalMatrixVectorMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CalculateDistribution(int rows, int proc_num, std::vector<int> &counts, std::vector<int> &displs);
};

}  // namespace salykina_a_horizontal_matrix_vector
