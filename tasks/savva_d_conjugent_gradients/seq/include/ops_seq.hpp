#pragma once

#include <vector>

#include "savva_d_conjugent_gradients/common/include/common.hpp"
#include "task/include/task.hpp"

namespace savva_d_conjugent_gradients {

class SavvaDConjugentGradientsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SavvaDConjugentGradientsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void UpdateXR(std::vector<double> &x, std::vector<double> &r, const std::vector<double> &p,
                       const std::vector<double> &ap, double alpha, int n);
  static void UpdateP(std::vector<double> &p, const std::vector<double> &r, double beta, int n);
  static void ComputeAp(const std::vector<double> &a, const std::vector<double> &p, std::vector<double> &ap, int n);
};

}  // namespace savva_d_conjugent_gradients
