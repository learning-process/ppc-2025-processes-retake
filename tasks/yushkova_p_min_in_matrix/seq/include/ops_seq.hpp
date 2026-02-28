#pragma once

#include "task/include/task.hpp"
#include "yushkova_p_min_in_matrix/common/include/common.hpp"

namespace yushkova_p_min_in_matrix {

class YushkovaPMinInMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit YushkovaPMinInMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace yushkova_p_min_in_matrix
