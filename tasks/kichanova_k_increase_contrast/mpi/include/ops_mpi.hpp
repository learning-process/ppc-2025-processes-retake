#pragma once

#include "kichanova_k_increase_contrast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kichanova_k_increase_contrast {

class KichanovaKIncreaseContrastMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit KichanovaKIncreaseContrastMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::tuple<int, int, int> CalculateRowsDistribution(int rank, int size, int height);
  std::array<uint8_t, 3> FindLocalMin(const Image &input, int start_row, int end_row, int width);
  std::array<uint8_t, 3> FindLocalMax(const Image &input, int start_row, int end_row, int width);
  std::tuple<std::array<float, 3>, std::array<bool, 3>> CalculateScaleFactors(const std::array<uint8_t, 3> &global_min,
                                                                              const std::array<uint8_t, 3> &global_max);
  std::vector<uint8_t> ProcessLocalRows(const Image &input, int start_row, int local_rows, int width,
                                        const std::array<uint8_t, 3> &global_min, const std::array<float, 3> &scale,
                                        const std::array<bool, 3> &need_scale);
};

}  // namespace kichanova_k_increase_contrast
