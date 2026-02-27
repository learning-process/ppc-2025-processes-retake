#pragma once

#include "salykina_a_sparse_matrix_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace salykina_a_sparse_matrix_mult {

class SalykinaASparseMatrixMultSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SalykinaASparseMatrixMultSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace salykina_a_sparse_matrix_mult
