#include <gtest/gtest.h>

#include "sokolov_k_shell_simple_merge/common/include/common.hpp"
#include "sokolov_k_shell_simple_merge/mpi/include/ops_mpi.hpp"
#include "sokolov_k_shell_simple_merge/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sokolov_k_shell_simple_merge {

class SokolovKShellSimpleMergePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 500000;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SokolovKShellSimpleMergePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SokolovKShellSimpleMergeMPI, SokolovKShellSimpleMergeSEQ>(
        PPC_SETTINGS_sokolov_k_shell_simple_merge);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SokolovKShellSimpleMergePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SokolovKShellSimpleMergePerfTests, kGtestValues, kPerfTestName);

}  // namespace sokolov_k_shell_simple_merge
