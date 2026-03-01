#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "klimov_m_shell_odd_even_merge/common/include/common.hpp"
#include "klimov_m_shell_odd_even_merge/mpi/include/ops_mpi.hpp"
#include "klimov_m_shell_odd_even_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace klimov_m_shell_odd_even_merge {

class ShellBatcherPerfTest : public ppc::util::BaseRunPerfTests<InputType, OutputType> {
 protected:
  void SetUp() override {
    perf_input_.resize(kDataSize);
    for (size_t i = 0; i < kDataSize; ++i) {
      perf_input_[i] = static_cast<int>(kDataSize - i);
    }
  }

  bool CheckTestOutputData(OutputType &out_data) final {
    return std::is_sorted(out_data.begin(), out_data.end());
  }

  InputType GetTestInputData() final {
    return perf_input_;
  }

 private:
  static constexpr size_t kDataSize = 3000000;
  InputType perf_input_;
};

namespace {

TEST_P(ShellBatcherPerfTest, MeasurePerformance) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InputType, ShellBatcherMPI, ShellBatcherSEQ>(
    PPC_SETTINGS_klimov_m_shell_odd_even_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kNameGen = ShellBatcherPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ShellBatcherPerformanceTests, ShellBatcherPerfTest, kGtestValues, kNameGen);

}  // namespace

}  // namespace klimov_m_shell_odd_even_merge
