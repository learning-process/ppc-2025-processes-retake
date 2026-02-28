#pragma once
#include "solonin_v_col_min_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace solonin_v_col_min_matrix {
class SoloninVMinMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() { return ppc::task::TypeOfTask::kSEQ; }
  explicit SoloninVMinMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace solonin_v_col_min_matrix
