#pragma once

#include "Nazarova_K_Gaussian_vert_scheme/common/include/common.hpp"

namespace nazarova_k_gaussian_vert_scheme_processes {

class NazarovaKGaussianVertSchemeMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NazarovaKGaussianVertSchemeMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Local vertical strip of the augmented matrix (n x local_cols)
  std::vector<double> local_aug_;
  int n_ = 0;
  int col_start_ = 0;
  int col_end_ = 0;
  int local_cols_ = 0;
};

}  // namespace nazarova_k_gaussian_vert_scheme_processes
