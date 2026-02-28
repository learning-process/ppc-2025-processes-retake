#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

class LuchnikovEGenerTransmFromAllToOneGatherPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const size_t kCount_ = 1000;
  InType input_data_{};

  void SetUp() override {
    input_data_.resize(kCount_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 1000);

    for (size_t i = 0; i < kCount_; ++i) {
      input_data_[i] = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    OutType expected = input_data_;
    std::sort(expected.begin(), expected.end());
    return expected == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuchnikovEGenerTransmFromAllToOneGatherPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LuchnikovEGenerTransmFromAllToOneGatherMPI,
                                                       LuchnikovEGenerTransmFromAllToOneGatherSEQ>(
    PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuchnikovEGenerTransmFromAllToOneGatherPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnikovEGenerTransmFromAllToOneGatherPerfTestProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
