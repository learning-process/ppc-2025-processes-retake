#pragma once

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "task/include/task.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnikovEMaxValInColOfMatSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LuchnikovEMaxValInColOfMatSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> local_result_;
  std::vector<std::vector<int>> input_copy_;
  int rows_ = 0;
  int cols_ = 0;
};

}  // namespace luchnikov_e_max_val_in_col_of_mat
