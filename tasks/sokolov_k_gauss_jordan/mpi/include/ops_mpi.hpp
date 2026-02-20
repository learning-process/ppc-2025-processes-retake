#pragma once

#include <vector>

#include "sokolov_k_gauss_jordan/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sokolov_k_gauss_jordan {

class SokolovKGaussJordanMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SokolovKGaussJordanMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  int n_{};
  std::vector<double> matrix_;
  std::vector<double> original_matrix_;
  std::vector<double> solution_;
};

}  // namespace sokolov_k_gauss_jordan
