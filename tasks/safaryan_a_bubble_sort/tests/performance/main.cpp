#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

#include "safaryan_a_bubble_sort/common/include/common.hpp"
#include "safaryan_a_bubble_sort/mpi/include/ops_mpi.hpp"
#include "safaryan_a_bubble_sort/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace safaryan_a_bubble_sort {
class SafaryanABubbleSortRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t size_params_ = 20000;
  InType input_data_;

  void SetUp() override {
    input_data_.resize(size_params_);
    for (size_t i = 0; i < size_params_; i++) {
      input_data_[i] = static_cast<int>(size_params_) - static_cast<int>(i);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::ranges::is_sorted(output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SafaryanABubbleSortRunPerfTestsProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SafaryanABubbleSortMPI, SafaryanABubbleSortSEQ>(
    PPC_SETTINGS_safaryan_a_bubble_sort);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SafaryanABubbleSortRunPerfTestsProcesses::CustomPerfTestName;

namespace safaryan_a_bubble_sort {

class SafaryanABubbleSortRunPerfTestsProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  ...
};

namespace {

INSTANTIATE_TEST_SUITE_P(RunModeTests, SafaryanABubbleSortRunPerfTestsProcesses, kGtestValues,
                         kPerfTestName);  // NOLINT(misc-use-anonymous-namespace)

}  // namespace

}  // namespace safaryan_a_bubble_sort
