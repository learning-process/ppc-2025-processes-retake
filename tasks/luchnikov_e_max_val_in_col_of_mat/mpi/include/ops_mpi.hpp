#pragma once

#include <utility>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnikovEMaxValInColOfMatMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LuchnikovEMaxValInColOfMatMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> RunSequential();
  std::vector<int> PrepareFlatMatrix();
  std::pair<std::vector<int>, std::vector<int>> CalculateDistribution();
  std::vector<int> ComputeLocalMax(const std::vector<int> &local_flat, int local_rows);

  std::vector<std::vector<int>> matrix_;
  std::vector<int> result_;
  int rows_ = 0;
  int cols_ = 0;
  int rank_ = 0;
  int size_ = 0;
};

}  // namespace luchnikov_e_max_val_in_col_of_mat
