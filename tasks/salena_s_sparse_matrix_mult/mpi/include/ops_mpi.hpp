#pragma once
#include "salena_s_sparse_matrix_mult/common/include/common.hpp"

namespace salena_s_sparse_matrix_mult {

class SparseMatrixMultMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SparseMatrixMultMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace salena_s_sparse_matrix_mult