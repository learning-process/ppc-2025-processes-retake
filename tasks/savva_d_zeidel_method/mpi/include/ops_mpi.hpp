#pragma once

#include <vector>

#include "savva_d_zeidel_method/common/include/common.hpp"
#include "task/include/task.hpp"

namespace savva_d_zeidel_method {

class SavvaDZeidelMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SavvaDZeidelMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void RunSeidelIterations(int n, int local_rows, int local_offset, const double *local_data_a,
                                  const double *local_data_b, std::vector<double> &x, const int *counts2,
                                  const int *displacements2);
};

}  // namespace savva_d_zeidel_method
