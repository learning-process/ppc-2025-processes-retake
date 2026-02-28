#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <random>

#include "denisov_a_min_val_row_matrix/common/include/common.hpp"
#include "denisov_a_min_val_row_matrix/mpi/include/ops_mpi.hpp"
#include "denisov_a_min_val_row_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace denisov_a_min_val_row_matrix {

class DenisovAMinValRowMatrixPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kRows = 5000;
  static constexpr int kCols = 5000;
  InType input_data_;
  OutType expected_output_data_;

  void SetUp() override {
    input_data_.resize(kRows);
    for (int i = 0; i < kRows; ++i) {
      input_data_[i].resize(kCols);
    }

    unsigned int seed = 4;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> distrib(-1000, 1000);
    input_data_.resize(kRows);
    for (int i = 0; i < kRows; ++i) {
      input_data_[i].resize(kCols);
      std::generate(input_data_[i].begin(), input_data_[i].end(), [&gen, &distrib]() { return distrib(gen); });
    }

    expected_output_data_.resize(kRows);
    for (int i = 0; i < kRows; ++i) {
      expected_output_data_[i] = *std::min_element(input_data_[i].begin(), input_data_[i].end());
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (ppc::util::IsUnderMpirun()) {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (rank != 0) {
        return true;
      }
    }
    return expected_output_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(DenisovAMinValRowMatrixPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, DenisovAMinValRowMatrixMPI, DenisovAMinValRowMatrixSEQ>(
    PPC_SETTINGS_denisov_a_min_val_row_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = DenisovAMinValRowMatrixPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, DenisovAMinValRowMatrixPerfTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace denisov_a_min_val_row_matrix
