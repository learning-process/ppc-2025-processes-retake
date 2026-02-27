#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "util/include/perf_test_util.hpp"
#include "zyuzin_n_sort_double_simple_merge/common/include/common.hpp"
#include "zyuzin_n_sort_double_simple_merge/mpi/include/ops_mpi.hpp"
#include "zyuzin_n_sort_double_simple_merge/seq/include/ops_seq.hpp"

namespace zyuzin_n_sort_double_simple_merge {

class ZyuzinNSortDoubleSimpleMergePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  const int k_count = 1000000;
  InType input_data;
  InType expected;

  void SetUp() override {
    input_data.resize(k_count);

    for (int i = 0; i < k_count; ++i) {
      input_data[i] = static_cast<double>((i * 37) % 10001) - 5000.0;
    }

    expected = input_data;
    std::ranges::sort(expected);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(k_count)) {
      return false;
    }
    for (std::size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected[i]) > 1e-12) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

TEST_P(ZyuzinNSortDoubleSimpleMergePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, ZyuzinNSortDoubleWithSimpleMergeMPI, ZyuzinNSortDoubleWithSimpleMergeSEQ>(
        PPC_SETTINGS_zyuzin_n_sort_double_simple_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ZyuzinNSortDoubleSimpleMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ZyuzinNSortDoubleSimpleMergePerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace zyuzin_n_sort_double_simple_merge
