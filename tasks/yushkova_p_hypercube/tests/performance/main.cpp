#include <gtest/gtest.h>

#include <cstdint>

#include "util/include/perf_test_util.hpp"
#include "yushkova_p_hypercube/common/include/common.hpp"
#include "yushkova_p_hypercube/mpi/include/ops_mpi.hpp"
#include "yushkova_p_hypercube/seq/include/ops_seq.hpp"

namespace yushkova_p_hypercube {
namespace {

OutType ReferenceEdges(InType n) {
  return static_cast<OutType>(n) * (static_cast<std::uint64_t>(1) << (n - 1));
}

}  // namespace

class YushkovaPHypercubePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  void SetUp() override {
    input_n_ = 25;
  }

 protected:
  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == ReferenceEdges(input_n_);
  }

  InType GetTestInputData() final {
    return input_n_;
  }

 private:
  InType input_n_{};
};

TEST_P(YushkovaPHypercubePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, YushkovaPHypercubeMPI, YushkovaPHypercubeSEQ>(
    PPC_SETTINGS_yushkova_p_hypercube);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = YushkovaPHypercubePerfTests::CustomPerfTestName;

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,misc-use-anonymous-namespace)
INSTANTIATE_TEST_SUITE_P(YushkovaPHypercubePerf, YushkovaPHypercubePerfTests, kGtestValues, kPerfTestName);
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,misc-use-anonymous-namespace)

}  // namespace

}  // namespace yushkova_p_hypercube
