#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "denisov_a_quick_sort_simple_merging/common/include/common.hpp"
#include "denisov_a_quick_sort_simple_merging/mpi/include/ops_mpi.hpp"
#include "denisov_a_quick_sort_simple_merging/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace denisov_a_quick_sort_simple_merging {

class DenisovAQuickSortMergePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 private:
  static constexpr size_t kDataCount = 1'000'000;
  InType buffer_;

  void SetUp() override {
    buffer_.resize(kDataCount);
    for (size_t idx = 0; idx < kDataCount; idx++) {
      buffer_[idx] = static_cast<int>(kDataCount - idx);
    }
  }

  bool CheckTestOutputData(OutType &result) final {
    if (result.size() != buffer_.size()) {
      return false;
    }

    std::vector<int> sorted_ref = buffer_;
    std::ranges::sort(sorted_ref);

    return result == sorted_ref && std::ranges::is_sorted(result);
  }

  InType GetTestInputData() final {
    return buffer_;
  }
};

TEST_P(DenisovAQuickSortMergePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, DenisovAQuickSortMergeMPI, DenisovAQuickSortMergeSEQ>(
    PPC_SETTINGS_denisov_a_quick_sort_simple_merging);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DenisovAQuickSortMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DenisovAQuickSortMergePerfTests, kGtestValues, kPerfTestName);

}  // namespace denisov_a_quick_sort_simple_merging
