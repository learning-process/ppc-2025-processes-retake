#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <random>

#include "denisov_a_ring/common/include/common.hpp"
#include "denisov_a_ring/mpi/include/ops_mpi.hpp"
#include "denisov_a_ring/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace denisov_a_ring {

class DenisovARingPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kVectorLength = 1'000'000;

  InType input_{};
  OutType expected_;

  void SetUp() override {
    int world_size = 1;

    if (ppc::util::IsUnderMpirun()) {
      MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    } else {
      world_size = ppc::util::GetNumProc();
    }

    input_.source = 0;
    input_.destination = (world_size > 1 ? world_size - 1 : 0);
    input_.data.resize(kVectorLength);

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> distrib(-10000, 10000);

    for (int i = 0; i < kVectorLength; ++i) {
      input_.data[static_cast<std::size_t>(i)] = distrib(rng);
    }

    expected_ = input_.data;
  }

  bool CheckTestOutputData(OutType &out) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    return out == expected_;
  }

  InType GetTestInputData() final {
    return input_;
  }
};

TEST_P(DenisovARingPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, RingTopologyMPI, RingTopologySEQ>(PPC_SETTINGS_denisov_a_ring);

const auto kValues = ppc::util::TupleToGTestValues(kPerfTasks);

const auto kNameGen = DenisovARingPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DenisovARingPerfTest, kValues, kNameGen);

}  // namespace denisov_a_ring
