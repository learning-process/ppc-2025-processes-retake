#include <gtest/gtest.h>

#include "tsarkov_k_monte_carlo_integration/common/include/common.hpp"
#include "tsarkov_k_monte_carlo_integration/mpi/include/ops_mpi.hpp"
#include "tsarkov_k_monte_carlo_integration/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace tsarkov_k_monte_carlo_integration {

class TsarkovKMonteCarloPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    input_data_ = InType{5, 500000, 123};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::isfinite(output_data) && (output_data > 0.0) && (output_data <= 1.0);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

TEST_P(TsarkovKMonteCarloPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, TsarkovKMonteCarloIntegrationMPI>(
    PPC_SETTINGS_tsarkov_k_monte_carlo_integration);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = TsarkovKMonteCarloPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, TsarkovKMonteCarloPerfTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace tsarkov_k_monte_carlo_integration
