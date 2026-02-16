#include <gtest/gtest.h>
#include <mpi.h>

#include "dergynov_s_hypercube/common/include/common.hpp"
#include "dergynov_s_hypercube/mpi/include/ops_mpi.hpp"
#include "dergynov_s_hypercube/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace dergynov_s_hypercube {

class DergynovSHypercubePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    input_ = {0, size - 1, kDataSize_};
  }

  bool CheckTestOutputData(OutType &out) final {
    return out == kDataSize_;
  }

  InType GetTestInputData() final {
    return input_;
  }

 private:
  const int kDataSize_ = 400;
  InType input_;
};

TEST_P(DergynovSHypercubePerfTests, RunPerf) {
  ExecuteTest(GetParam());
}

const auto kAllTasks = ppc::util::MakeAllPerfTasks<InType, DergynovSHypercubeMPI, DergynovSHypercubeSEQ>(
    PPC_SETTINGS_dergynov_s_hypercube);

const auto kValues = ppc::util::TupleToGTestValues(kAllTasks);

INSTANTIATE_TEST_SUITE_P(PerfTests, DergynovSHypercubePerfTests, kValues,
                         DergynovSHypercubePerfTests::CustomPerfTestName);

}  // namespace dergynov_s_hypercube
