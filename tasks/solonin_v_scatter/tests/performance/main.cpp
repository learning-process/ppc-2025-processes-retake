#include <gtest/gtest.h>

#include <cstddef>
#include <numeric>
#include <tuple>
#include <vector>

#include "solonin_v_scatter/common/include/common.hpp"
#include "solonin_v_scatter/mpi/include/ops_mpi.hpp"
#include "solonin_v_scatter/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace solonin_v_scatter {

class SoloninVScatterPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kSendCount = 1000000;
  static constexpr int kMaxProcs = 4;

  std::vector<int> buf_;
  int send_count_{kSendCount};
  int root_{0};

  void SetUp() override {
    buf_.resize(kSendCount * kMaxProcs);
    std::iota(buf_.begin(), buf_.end(), 0);
  }

  bool CheckTestOutputData(OutType &out) final {
    return static_cast<int>(out.size()) == send_count_;
  }

  InType GetTestInputData() final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == root_) {
      return std::make_tuple(buf_, send_count_, root_);
    }
    return std::make_tuple(std::vector<int>(), send_count_, root_);
  }
};

TEST_P(SoloninVScatterPerfTests, RunPerfModes) { ExecuteTest(GetParam()); }

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SoloninVScatterMPI, SoloninVScatterSEQ>(
    PPC_SETTINGS_solonin_v_scatter);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = SoloninVScatterPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SoloninVScatterPerfTests, kGtestValues, kPerfTestName);

}  // namespace solonin_v_scatter
