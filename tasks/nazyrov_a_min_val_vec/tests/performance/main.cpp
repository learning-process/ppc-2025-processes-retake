#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <random>

#include "nazyrov_a_min_val_vec/common/include/common.hpp"
#include "nazyrov_a_min_val_vec/mpi/include/ops_mpi.hpp"
#include "nazyrov_a_min_val_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace nazyrov_a_min_val_vec {

class NazyrovAMinValVecPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 10000000;
  InType input_data_;
  OutType expected_{0};

  void SetUp() override {
    input_data_.resize(kDataSize);
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(-1000000, 1000000);
    for (std::size_t i = 0; i < input_data_.size(); ++i) {
      input_data_[i] = dist(gen);
    }
    expected_ = *std::ranges::min_element(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    return output_data == expected_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NazyrovAMinValVecPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MinValVecMPI, MinValVecSEQ>(PPC_SETTINGS_nazyrov_a_min_val_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NazyrovAMinValVecPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NazyrovAMinValVecPerfTest, kGtestValues, kPerfTestName);

}  // namespace nazyrov_a_min_val_vec
