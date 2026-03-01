#pragma once

#include <vector>

#include "sokolov_k_min_val_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace k_sokolov_min_val_matrix {

class SokolovKMinValMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SokolovKMinValMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<int> matrix_;
  int rows_ = 0;
  int cols_ = 0;
  int min_val_ = 0;
};

}  // namespace k_sokolov_min_val_matrix
