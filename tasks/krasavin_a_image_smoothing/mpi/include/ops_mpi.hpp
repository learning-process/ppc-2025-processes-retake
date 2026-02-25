#pragma once

#include <vector>

#include "krasavin_a_image_smoothing/common/include/common.hpp"
#include "task/include/task.hpp"

namespace krasavin_a_image_smoothing {

class KrasavinAImageSmoothingMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KrasavinAImageSmoothingMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<std::vector<float>> gaussian_kernel_;
};

}  // namespace krasavin_a_image_smoothing
