#pragma once

#include <vector>

#include "denisov_a_min_val_row_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace denisov_a_min_val_row_matrix {

class DenisovAMinValRowMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit DenisovAMinValRowMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void RunMaster(int rows, int cols, int base_rows, int extra_rows, int local_rows, std::vector<int> &local_matrix,
                 std::vector<int> &local_min);
  static void RunWorker(int cols, int local_rows, std::vector<int> &local_matrix, std::vector<int> &local_min);
};

}  // namespace denisov_a_min_val_row_matrix
