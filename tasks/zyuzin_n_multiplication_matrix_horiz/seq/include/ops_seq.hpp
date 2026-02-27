#pragma once

#include "task/include/task.hpp"
#include "zyuzin_n_multiplication_matrix_horiz/common/include/common.hpp"

namespace zyuzin_n_multiplication_matrix_horiz {

class ZyuzinNMultiplicationMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZyuzinNMultiplicationMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zyuzin_n_multiplication_matrix_horiz
