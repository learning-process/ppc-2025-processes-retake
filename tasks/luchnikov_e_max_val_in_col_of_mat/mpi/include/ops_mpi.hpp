// [file name]: mpi/include/ops_mpi.hpp
#pragma once

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnilkovEMaxValInColOfMatMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LuchnilkovEMaxValInColOfMatMPI(const InType &in);

  int GetRank() const {
    return rank_;
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int rank_ = 0;
  int size_ = 1;
};

}  // namespace luchnikov_e_max_val_in_col_of_mat
