#pragma once

#include "kaur_a_multy_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kaur_a_multy_matrix {

class KaurAMultyMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit KaurAMultyMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void TransposeMatrix(const SparseMatrixCCS &a, SparseMatrixCCS &at);
  static void MultiplyMatrices(const SparseMatrixCCS &a, const SparseMatrixCCS &b, SparseMatrixCCS &c);
};

}  // namespace kaur_a_multy_matrix