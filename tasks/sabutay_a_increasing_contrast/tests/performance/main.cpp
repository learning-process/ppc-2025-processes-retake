#include <gtest/gtest.h>

#include <cstddef>

#include "sabutay_a_increasing_contrast/common/include/common.hpp"
#include "sabutay_a_increasing_contrast/mpi/include/ops_mpi.hpp"
#include "sabutay_a_increasing_contrast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sabutay_a_increasing_contrast {

class SabutayAIncreaseContrastPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t kPixelsCount_ = static_cast<size_t>(8192) * 8192;  // 8к изображение
  InType input_data_;

  void SetUp() override {
    input_data_.resize(kPixelsCount_);
    for (size_t i = 0; i < kPixelsCount_; i++) {
      // повторяющийся блок от 100 до 150
      input_data_[i] = static_cast<unsigned char>(100 + (i % 51));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // считать значения - безумие, поэтому проверяем осмысленность результата
    return !output_data.empty() && output_data.size() == kPixelsCount_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SabutayAIncreaseContrastPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SabutayAIncreaseContrastMPI, SabutayAIncreaseContrastSEQ>(
        PPC_SETTINGS_sabutay_a_increasing_contrast);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SabutayAIncreaseContrastPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SabutayAIncreaseContrastPerfTests, kGtestValues, kPerfTestName);

}  // namespace sabutay_a_increasing_contrast
