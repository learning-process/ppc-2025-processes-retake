#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <random>

#include "nazyrov_a_broadcast/common/include/common.hpp"
#include "nazyrov_a_broadcast/mpi/include/ops_mpi.hpp"
#include "nazyrov_a_broadcast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace nazyrov_a_broadcast {

class NazyrovABroadcastPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 1000000;
  InType input_data_{};
  OutType expected_output_;

  void SetUp() override {
    input_data_.root = 0;
    input_data_.data.resize(kDataSize);

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(-10000, 10000);
    for (std::size_t i = 0; i < input_data_.data.size(); ++i) {
      input_data_.data[i] = dist(gen);
    }

    expected_output_ = input_data_.data;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NazyrovABroadcastPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BroadcastMPI, BroadcastSEQ>(PPC_SETTINGS_nazyrov_a_broadcast);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NazyrovABroadcastPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NazyrovABroadcastPerfTest, kGtestValues, kPerfTestName);

}  // namespace nazyrov_a_broadcast
