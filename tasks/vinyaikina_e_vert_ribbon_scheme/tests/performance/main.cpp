#include <gtest/gtest.h>

#include "util/include/perf_test_util.hpp"
#include "vinyaikina_e_vert_ribbon_scheme/common/include/common.hpp"
#include "vinyaikina_e_vert_ribbon_scheme/mpi/include/ops_mpi.hpp"
#include "vinyaikina_e_vert_ribbon_scheme/seq/include/ops_seq.hpp"

namespace vinyaikina_e_vert_ribbon_scheme {

class VinyaikinaEVertRibbonSchemePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 4000;
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

TEST_P(VinyaikinaEVertRibbonSchemePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VinyaikinaEVertRibbonSchemeMPI, VinyaikinaEVertRibbonSchemeSEQ>(
        PPC_SETTINGS_vinyaikina_e_vert_ribbon_scheme);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VinyaikinaEVertRibbonSchemePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, VinyaikinaEVertRibbonSchemePerfTests, kGtestValues, kPerfTestName);

}  // namespace vinyaikina_e_vert_ribbon_scheme
