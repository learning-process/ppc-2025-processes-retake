#pragma once

#include <vector>

#include "savva_d_conjugent_gradients/common/include/common.hpp"
#include "task/include/task.hpp"

namespace savva_d_conjugent_gradients {

class SavvaDConjugentGradientsMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SavvaDConjugentGradientsMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void UpdateXR(std::vector<double> &x, std::vector<double> &r, const std::vector<double> &p,
                       const std::vector<double> &global_ap, double alpha, int n);
  static std::vector<double> ComputeLocalAp(int n, int local_rows, const std::vector<double> &local_a,
                                            const std::vector<double> &p);
  static void RunCGIterations(int n, int local_rows, int local_offset, std::vector<double> &r,
                              const std::vector<double> &local_a, std::vector<double> &vector_x,
                              const std::vector<int> &counts, const std::vector<int> &displs);
};

}  // namespace savva_d_conjugent_gradients
