#pragma once
#include <vector>
#include "solonin_v_sparse_matrix_crs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace solonin_v_sparse_matrix_crs {

class SoloninVSparseMulCRSMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() { return ppc::task::TypeOfTask::kMPI; }
  explicit SoloninVSparseMulCRSMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool RunSequential();
  bool BroadcastSizes(int &rows_a, int &cols_a, int &cols_b);
  void BroadcastB();
  void DistributeA();
  void SendAToRank(int dest);
  void ReceiveAFromRoot();
  void ComputeLocal();
  void ProcessLocalRow(int local_idx, std::vector<double> &row_vals, std::vector<int> &row_cols);
  void GatherResults();
  void CollectFromRank(int src, std::vector<std::vector<double>> &all_vals,
                       std::vector<std::vector<int>> &all_cols);
  void AssembleResult(std::vector<std::vector<double>> &all_vals,
                      std::vector<std::vector<int>> &all_cols);
  static void SortRow(std::vector<double> &rv, std::vector<int> &rc);

  std::vector<double> vals_a_, vals_b_, vals_c_;
  std::vector<int> cols_a_, cols_b_, cols_c_;
  std::vector<int> ptr_a_, ptr_b_, ptr_c_;
  int rows_a_{0}, cols_a_count_{0}, cols_b_count_{0};

  std::vector<int> local_row_ids_;
  std::vector<double> local_vals_a_, local_vals_c_;
  std::vector<int> local_cols_a_, local_cols_c_;
  std::vector<int> local_ptr_a_, local_ptr_c_;

  int rank_{0}, world_size_{1};
};

}  // namespace solonin_v_sparse_matrix_crs
