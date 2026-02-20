#pragma once

#include <vector>

#include "kamaletdinov_r_gauss_vertical_scheme/kamaletdinov_r_gauss_vertical_scheme/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kamaletdinov_r_gauss_vertical_scheme {

class KamaletdinovRGaussVerticalSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KamaletdinovRGaussVerticalSchemeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  int FindPivotRow(int k, int cols);
  void SwapRows(int row1, int row2, int cols);
  void EliminateColumn(int k, int cols);
  void SynchronizeRow(int k, int row, int cols);
  void BackSubstitution();

  void DistributeMatrixByStripes();
  void ReceiveMatrixStripe();
  void SynchronizeMatrixByStripes();
  void ExchangeStripesForRow(int row);
  void ExchangeStripeWithProcess(int row, int proc);

  int n_{0};
  std::vector<double> extended_matrix_;
  std::vector<double> solution_;
  int rank_{0};
  int size_{0};
};

}  // namespace kamaletdinov_r_gauss_vertical_scheme
