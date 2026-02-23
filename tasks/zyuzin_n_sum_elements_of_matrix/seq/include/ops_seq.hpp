#pragma once

#include "task/include/task.hpp"
#include "zyuzin_n_sum_elements_of_matrix/common/include/common.hpp"

namespace zyuzin_n_sum_elements_of_matrix {

class ZyuzinNSumElementsOfMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit ZyuzinNSumElementsOfMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zyuzin_n_sum_elements_of_matrix
