#pragma once

#include <vector>

#include "rysev_m_matrix_multiple/common/include/common.hpp"
#include "task/include/task.hpp"

namespace rysev_m_matrix_multiple {

class RysevMMatrMulMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit RysevMMatrMulMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ComputeDistribution(std::vector<int> &send_counts, std::vector<int> &displs) const;
  void PrepareLocalBuffers(const std::vector<int> &send_counts);
  void LocalMultiply();
  void ComputeGatherParams(std::vector<int> &recv_counts, std::vector<int> &recv_displs) const;

  std::vector<int> A_;
  std::vector<int> B_;
  std::vector<int> C_;
  int size_ = 0;
  int rank_ = 0;
  int num_procs_ = 0;
  std::vector<int> local_A_;
  std::vector<int> local_C_;
  int local_rows_ = 0;
};

}  // namespace rysev_m_matrix_multiple
