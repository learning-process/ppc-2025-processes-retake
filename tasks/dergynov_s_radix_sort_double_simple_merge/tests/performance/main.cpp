#include <gtest/gtest.h>

#include <cstddef>

#include "dergynov_s_radix_sort_double_simple_merge/common/include/common.hpp"
#include "dergynov_s_radix_sort_double_simple_merge/mpi/include/ops_mpi.hpp"
#include "dergynov_s_radix_sort_double_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dergynov_s_radix_sort_double_simple_merge {

class DergynovRadixSortPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  std::size_t kCount_ = 1000000;

  void SetUp() override {
    test_data_.resize(kCount_);
    for (std::size_t i = 0; i < kCount_; ++i) {
      auto value = static_cast<double>(kCount_ - i);
      if (i % 3 == 0) {
        value = -value;
      }
      if (i % 7 == 0) {
        value += 0.25;
      }
      test_data_[i] = value;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (std::get<1>(output_data) == 0) {
      for (std::size_t i = 1; i < std::get<0>(output_data).size(); ++i) {
        if (std::get<0>(output_data)[i] < std::get<0>(output_data)[i - 1]) {
          return false;
        }
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return test_data_;
  }

 private:
  InType test_data_;
};

TEST_P(DergynovRadixSortPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, DergynovSRadixSortDoubleSimpleMergeMPI, DergynovSRadixSortDoubleSimpleMergeSEQ>(
        PPC_SETTINGS_dergynov_s_radix_sort_double_simple_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DergynovRadixSortPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DergynovRadixSortPerfTests, kGtestValues, kPerfTestName);

}  // namespace dergynov_s_radix_sort_double_simple_merge
