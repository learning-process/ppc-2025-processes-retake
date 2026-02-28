#pragma once

#include <vector>

#include "nozdrin_a_mult_matr_cannon/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nozdrin_a_mult_matr_cannon {

class NozdrinAMultMatrCannonMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NozdrinAMultMatrCannonMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void LocalMatrixMultiply(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                                  int n);
};

}  // namespace nozdrin_a_mult_matr_cannon
