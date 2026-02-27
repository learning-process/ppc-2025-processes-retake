#pragma once
#include <vector>

#include "salykina_a_sparse_matrix_mult/common/include/common.hpp"
#include "task/include/task.hpp"

namespace salykina_a_sparse_matrix_mult {

class SalykinaASparseMatrixMultMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SalykinaASparseMatrixMultMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  void MultiplyHorizontalScheme();
  void GatherResults(const std::vector<double> &local_values, const std::vector<int> &local_col_indices,
                     const std::vector<int> &local_row_ptr, int local_start_row, int local_num_rows);
  static void BroadcastMatrixB(int rank, int size, const SparseMatrixCRS &b, std::vector<double> &b_values,
                               std::vector<int> &b_col_indices, std::vector<int> &b_row_ptr, int &b_num_rows,
                               int &b_num_cols);
  static void ComputeLocalRows(const SparseMatrixCRS &a, const std::vector<double> &b_values,
                               const std::vector<int> &b_col_indices, const std::vector<int> &b_row_ptr, int b_num_cols,
                               int local_start_row, int local_num_rows, std::vector<double> &local_values,
                               std::vector<int> &local_col_indices, std::vector<int> &local_row_ptr);
  void AssembleResultsOnRank0(int rank, int size, const std::vector<double> &local_values,
                              const std::vector<int> &local_col_indices, const std::vector<int> &local_row_ptr,
                              int local_num_rows, const std::vector<int> &nnz_counts,
                              const std::vector<int> &row_counts);
};
}  // namespace salykina_a_sparse_matrix_mult
