#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "klimov_m_lett_count/common/include/common.hpp"
#include "klimov_m_lett_count/mpi/include/ops_mpi.hpp"
#include "klimov_m_lett_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace klimov_m_lett_count {

class KlimovMLettCountPerfTest : public ppc::util::BaseRunPerfTests<InputType, OutputType> {
  static constexpr size_t kRepeatFactor = 10'000'000;
  static constexpr const char *kPattern = "abcdefgh";

  InputType large_string_;
  OutputType expected_letters_ = 0;

 protected:
  void SetUp() override {
    large_string_.reserve(kRepeatFactor * std::char_traits<char>::length(kPattern));
    for (size_t i = 0; i < kRepeatFactor; ++i) {
      large_string_ += kPattern;
    }
    expected_letters_ = static_cast<OutputType>(kRepeatFactor * std::char_traits<char>::length(kPattern));
  }

  bool CheckTestOutputData(OutputType &output) final {
    return expected_letters_ == output;
  }

  InputType GetTestInputData() final {
    return large_string_;
  }
};

namespace {

TEST_P(KlimovMLettCountPerfTest, MeasurePerformance) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InputType, KlimovMLettCountMPI, KlimovMLettCountSEQ>(
    "tasks/klimov_m_lett_count/settings.json");

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kNameGen = KlimovMLettCountPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(KlimovLettCountPerformance, KlimovMLettCountPerfTest, kGtestValues, kNameGen);

}  // namespace

}  // namespace klimov_m_lett_count
