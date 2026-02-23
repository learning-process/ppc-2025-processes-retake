#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kichanova_k_increase_contrast/common/include/common.hpp"
#include "kichanova_k_increase_contrast/mpi/include/ops_mpi.hpp"
#include "kichanova_k_increase_contrast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace kichanova_k_increase_contrast {

class KichanovaKIncreaseContrastPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kWidth_ = 6144;
  const int kHeight_ = 4096;

  InType input_data_;

  void SetUp() override {
    input_data_.width = kWidth_;
    input_data_.height = kHeight_;
    input_data_.channels = 3;
    input_data_.pixels.resize(static_cast<size_t>(kWidth_) * kHeight_ * 3);

    for (int row = 0; row < kHeight_; ++row) {
      for (int col = 0; col < kWidth_; ++col) {
        size_t idx = (static_cast<size_t>(row) * kWidth_ + col) * 3;
        input_data_.pixels[idx] = static_cast<uint8_t>((col * 255) / kWidth_);
        input_data_.pixels[idx + 1] = static_cast<uint8_t>((row * 255) / kHeight_);
        input_data_.pixels[idx + 2] = 128;
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.pixels.empty()) {
      return false;
    }

    return output_data.width == kWidth_ && output_data.height == kHeight_ &&
           output_data.pixels.size() == static_cast<size_t>(kWidth_) * kHeight_ * 3;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(KichanovaKIncreaseContrastPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KichanovaKIncreaseContrastMPI, KichanovaKIncreaseContrastSEQ>(
        PPC_SETTINGS_kichanova_k_increase_contrast);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = KichanovaKIncreaseContrastPerfTests::CustomPerfTestName;

namespace {
INSTANTIATE_TEST_SUITE_P(RunModeTests, KichanovaKIncreaseContrastPerfTests, kGtestValues, kPerfTestName);
}  // namespace

}  // namespace kichanova_k_increase_contrast
