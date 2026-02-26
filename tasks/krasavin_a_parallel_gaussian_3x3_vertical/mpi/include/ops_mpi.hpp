#pragma once

#include "krasavin_a_parallel_gaussian_3x3_vertical/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krasavin_a_parallel_gaussian_3x3_vertical {

class KrasavinAParallelGaussian3x3VerticalMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KrasavinAParallelGaussian3x3VerticalMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace krasavin_a_parallel_gaussian_3x3_vertical