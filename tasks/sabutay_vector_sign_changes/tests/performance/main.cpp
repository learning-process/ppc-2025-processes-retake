#include <gtest/gtest.h>

#include <string>

#include "sabutay_vector_sign_changes/common/include/common.hpp"
#include "sabutay_vector_sign_changes/mpi/include/ops_mpi.hpp"
#include "sabutay_vector_sign_changes/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace sabutay_vector_sign_changes {

class SabutayVectorSignChangesRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    input_data_ = 10000000;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    (void)output_data;
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SabutayVectorSignChangesRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SabutayVectorSignChangesMPI, SabutayVectorSignChangesSEQ>(
        PPC_SETTINGS_example_processes);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

namespace {

using ParamType = SabutayVectorSignChangesRunPerfTestProcesses::ParamType;

::testing::internal::ParamGenerator<ParamType> RastvorovPerfEvalGenerator() {
  return kGtestValues;
}

std::string RastvorovPerfEvalGenerateName(const ::testing::TestParamInfo<ParamType> &info) {
  return SabutayVectorSignChangesRunPerfTestProcesses::CustomPerfTestName(info);
}

const int kRastvorovPerfDummy =
    ::testing::UnitTest::GetInstance()
        ->parameterized_test_registry()
        .GetTestSuitePatternHolder<SabutayVectorSignChangesRunPerfTestProcesses>(
            "SabutayVectorSignChangesRunPerfTestProcesses", ::testing::internal::CodeLocation(__FILE__, __LINE__))
        ->AddTestSuiteInstantiation("RunModeTests", &RastvorovPerfEvalGenerator, &RastvorovPerfEvalGenerateName,
                                    __FILE__, __LINE__);

}  // namespace

}  // namespace sabutay_vector_sign_changes
