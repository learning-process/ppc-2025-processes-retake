#include <gtest/gtest.h>

#include <cstddef>
#include <tuple>
#include <vector>

#include "safaryan_a_sum_matrix/common/include/common.hpp"
#include "safaryan_a_sum_matrix/mpi/include/ops_mpi.hpp"
#include "safaryan_a_sum_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace safaryan_a_sum_matrix {

class SafaryanASumMatrixPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  // ВАЖНО: 10000x10000 = ~400MB только под матрицу.
  // На CI это легко приводит к падению MPI процесса.
  const int kMatrixRows_ = 2000;
  const int kMatrixCols_ = 2000;

  InType input_data_;
  OutType expected_result_;

  void SetUp() override {
    std::vector<int> matrix_data(static_cast<size_t>(kMatrixRows_) * static_cast<size_t>(kMatrixCols_));
    for (int i = 0; i < kMatrixRows_ * kMatrixCols_; ++i) {
      matrix_data[i] = (i % 100) + 1;
    }

    input_data_ = std::make_tuple(matrix_data, kMatrixRows_, kMatrixCols_);

    expected_result_.resize(kMatrixRows_);
    for (int i = 0; i < kMatrixRows_; ++i) {
      int sum = 0;
      for (int j = 0; j < kMatrixCols_; ++j) {
        sum += matrix_data[(i * kMatrixCols_) + j];
      }
      expected_result_[i] = sum;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SafaryanASumMatrixPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, SafaryanASumMatrixMPI, SafaryanASumMatrixSEQ>(
    PPC_SETTINGS_safaryan_a_sum_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SafaryanASumMatrixPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SafaryanASumMatrixPerfTests, kGtestValues, kPerfTestName);

}  // namespace safaryan_a_sum_matrix
