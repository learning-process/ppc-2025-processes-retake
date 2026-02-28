#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

#include "util/include/perf_test_util.hpp"
#include "yusupkina_m_mnog_integ_monte_carlo/common/include/common.hpp"
#include "yusupkina_m_mnog_integ_monte_carlo/mpi/include/ops_mpi.hpp"
#include "yusupkina_m_mnog_integ_monte_carlo/seq/include/ops_seq.hpp"

namespace yusupkina_m_mnog_integ_monte_carlo {

class YusupkinaMMnogIntegMonteCarloPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int64_t kPointsLarge_ = 10000000;

  InType input_data_;
  const OutType expected_output_ = 0.25;

  void SetUp() override {
    double x_min = 0.0;
    double x_max = 1.0;
    double y_min = 0.0;
    double y_max = 1.0;

    auto f = [](double x, double y) { return x * y; };
    input_data_ = InputData(x_min, x_max, y_min, y_max, f, kPointsLarge_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    const double epsilon = 0.05;
    return std::abs(output_data - expected_output_) <= epsilon * std::abs(expected_output_);
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(YusupkinaMMnogIntegMonteCarloPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, YusupkinaMMnogIntegMonteCarloMPI, YusupkinaMMnogIntegMonteCarloSEQ>(
        PPC_SETTINGS_yusupkina_m_mnog_integ_monte_carlo);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = YusupkinaMMnogIntegMonteCarloPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, YusupkinaMMnogIntegMonteCarloPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace yusupkina_m_mnog_integ_monte_carlo
