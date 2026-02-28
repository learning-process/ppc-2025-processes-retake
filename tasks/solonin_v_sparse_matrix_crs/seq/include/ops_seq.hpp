#pragma once
#include <vector>
#include "solonin_v_sparse_matrix_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace solonin_v_sparse_matrix_crs {

class SoloninVSparseMulCRSSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() { return ppc::task::TypeOfTask::kSEQ; }
  explicit SoloninVSparseMulCRSSEQ(InType in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] bool ValidateA() const;
  [[nodiscard]] bool ValidateB() const;
  void MultiplyRow(int row_idx, std::vector<double> &row_vals, std::vector<int> &row_cols);

  InType input_;
  std::vector<double> vals_a_;
  std::vector<int> cols_a_;
  std::vector<int> ptr_a_;
  int rows_a_;
  int cols_a_count_;
  std::vector<double> vals_b_;
  std::vector<int> cols_b_;
  std::vector<int> ptr_b_;
  int cols_b_count_;
  std::vector<double> vals_c_;
  std::vector<int> cols_c_;
  std::vector<int> ptr_c_;
};

}  // namespace solonin_v_sparse_matrix_crs
