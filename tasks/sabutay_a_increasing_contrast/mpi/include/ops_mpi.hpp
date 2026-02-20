#pragma once

#include <vector>

#include "sabutay_a_increasing_contrast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sabutay_a_increasing_contrast {

class SabutayAIncreaseContrastMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SabutayAIncreaseContrastMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  std::vector<unsigned char> ScatterInputData(int rank, int size, int data_len);
  static void FindGlobalMinMax(const std::vector<unsigned char> &proc_part, unsigned char *data_min,
                               unsigned char *data_max);
  static std::vector<unsigned char> ApplyContrast(const std::vector<unsigned char> &proc_part, unsigned char data_min,
                                                  unsigned char data_max);
};

}  // namespace sabutay_a_increasing_contrast
