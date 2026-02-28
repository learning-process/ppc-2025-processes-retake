#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "krasavin_a_max_neighbor_diff/common/include/common.hpp"
#include "krasavin_a_max_neighbor_diff/mpi/include/ops_mpi.hpp"
#include "krasavin_a_max_neighbor_diff/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace krasavin_a_max_neighbor_diff {

class KrasavinAMaxNeighborDiffPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int k_vector_size = 200000000;

    input_data_.resize(k_vector_size);

    std::random_device random;
    std::mt19937 gen(random());
    std::uniform_int_distribution<int> small_dist(0, 100);
    std::uniform_int_distribution<int> large_dist(1000, 10000);

    for (int i = 0; i < k_vector_size; i++) {
      if (i % 2 == 0) {
        input_data_[i] = small_dist(gen);
      } else {
        input_data_[i] = large_dist(gen);
      }
    }

    expected_max_diff_ = 0;
    for (int i = 0; i < k_vector_size - 1; i++) {
      int diff = std::abs(input_data_[i + 1] - input_data_[i]);
      expected_max_diff_ = std::max(diff, expected_max_diff_);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data >= expected_max_diff_ && output_data <= expected_max_diff_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  int expected_max_diff_ = 0;
};

TEST_P(KrasavinAMaxNeighborDiffPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, KrasavinAMaxNeighborDiffMPI, KrasavinAMaxNeighborDiffSEQ>(
        PPC_SETTINGS_krasavin_a_max_neighbor_diff);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = KrasavinAMaxNeighborDiffPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerfTests, KrasavinAMaxNeighborDiffPerfTests, kGtestValues, kPerfTestName);

}  // namespace krasavin_a_max_neighbor_diff
