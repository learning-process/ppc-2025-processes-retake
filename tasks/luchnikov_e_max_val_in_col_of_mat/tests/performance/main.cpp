#include <gtest/gtest.h>

#include <climits>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnikovEMaxValInColOfMatPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kDim_ = 150;
  InType input_data_{};
  OutType expected_output_{};

  void SetUp() override {
    input_data_.resize(kDim_, std::vector<int>(kDim_));

    for (int i = 0; i < kDim_; ++i) {
      for (int j = 0; j < kDim_; ++j) {
        input_data_[i][j] = ((i + 1) * 31 + (j + 1) * 37) % 5000;
      }
    }

    expected_output_.resize(kDim_, INT_MIN);
    for (int j = 0; j < kDim_; ++j) {
      for (int i = 0; i < kDim_; ++i) {
        if (input_data_[i][j] > expected_output_[j]) {
          expected_output_[j] = input_data_[i][j];
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LuchnikovEMaxValInColOfMatPerfTest, PerformanceRun) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuchnikovEMaxValInColOfMatMPI, LuchnikovEMaxValInColOfMatSEQ>(
        PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = LuchnikovEMaxValInColOfMatPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerformanceTests, LuchnikovEMaxValInColOfMatPerfTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_max_val_in_col_of_mat
