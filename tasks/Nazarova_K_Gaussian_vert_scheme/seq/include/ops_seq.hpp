#pragma once

#include "Nazarova_K_Gaussian_vert_scheme/common/include/common.hpp"

namespace nazarova_k_gaussian_vert_scheme_processes {

class NazarovaKGaussianVertSchemeSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit NazarovaKGaussianVertSchemeSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<double> aug_;
  int n_ = 0;
};

}  // namespace nazarova_k_gaussian_vert_scheme_processes
