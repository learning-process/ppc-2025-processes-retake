#include <gtest/gtest.h>

#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnikovEMaxValInColOfMatPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kRows_ = 500;
  const int kCols_ = 500;
  InType input_data_{};
  OutType expected_output_{};

 protected:
  void SetUp() override {
    input_data_.resize(kRows_, std::vector<int>(kCols_));
    for (int i = 0; i < kRows_; ++i) {
      for (int j = 0; j < kCols_; ++j) {
        input_data_[i][j] = (i * kCols_ + j) % 1000;
      }
    }

    expected_output_.resize(kCols_);
    for (int j = 0; j < kCols_; ++j) {
      int max_val = input_data_[0][j];
      for (int i = 1; i < kRows_; ++i) {
        if (input_data_[i][j] > max_val) {
          max_val = input_data_[i][j];
        }
      }
      expected_output_[j] = max_val;
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

TEST_P(LuchnikovEMaxValInColOfMatPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuchnikovEMaxValInColOfMatMPI, LuchnikovEMaxValInColOfMatSEQ>(
        PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ppc::util::BaseRunPerfTests<InType, OutType>::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnikovEMaxValInColOfMatPerfTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_max_val_in_col_of_mat
