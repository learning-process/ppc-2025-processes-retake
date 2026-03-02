#include <gtest/gtest.h>

#include "luchnikov_e_gener_transm_from_all_to_one_gather/common/include/common.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_gener_transm_from_all_to_one_gather/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_gener_transm_from_all_to_one_gather {

class LuchnikovEGenerTransformFromAllToOneGatherPerfTestProcesses
    : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 protected:
  void SetUp() override {
    input_data_ = kCount_;
  }

 private:
  const int kCount_ = 100;
  InType input_data_{};
};

namespace {

TEST_P(LuchnikovEGenerTransformFromAllToOneGatherPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, LuchnikovEGenerTransformFromAllToOneGatherMPI,
                                                       LuchnikovEGenerTransformFromAllToOneGatherSEQ>(
    PPC_SETTINGS_luchnikov_e_gener_transm_from_all_to_one_gather);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuchnikovEGenerTransformFromAllToOneGatherPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnikovEGenerTransformFromAllToOneGatherPerfTestProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_gener_transm_from_all_to_one_gather
