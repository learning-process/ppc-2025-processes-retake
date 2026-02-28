#pragma once

#include "denisov_a_min_val_row_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace denisov_a_min_val_row_matrix {

class DenisovAMinValRowMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit DenisovAMinValRowMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace denisov_a_min_val_row_matrix
