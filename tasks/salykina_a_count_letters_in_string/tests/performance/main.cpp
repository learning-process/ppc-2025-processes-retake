#include <gtest/gtest.h>

#include "salykina_a_count_letters_in_string/common/include/common.hpp"
#include "salykina_a_count_letters_in_string/mpi/include/ops_mpi.hpp"
#include "salykina_a_count_letters_in_string/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace salykina_a_count_letters_in_string {

class SalykinaARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 50000000;
  InType input_data_;

  void SetUp() override {
    input_data_ = "";
    for (int i = 0; i < kCount_; i++) {
      input_data_ += static_cast<char>('a' + (i % 26));

      if (i % 10 == 0) {
        input_data_ += "123!@#";
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == kCount_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

namespace {

TEST_P(SalykinaARunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SalykinaACountLettersMPI, SalykinaACountLettersSEQ>(
    PPC_SETTINGS_salykina_a_count_letters_in_string);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = SalykinaARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SalykinaARunPerfTestProcesses, kGtestValues, kPerfTestName);
}  // namespace

}  // namespace salykina_a_count_letters_in_string
