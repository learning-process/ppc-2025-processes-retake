#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <limits>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnikovEMaxValInColOfMatPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 100;
  InType input_data_;
  OutType expected_output_;

  void SetUp() override {
    input_data_ = GenerateLargeMatrix(kMatrixSize_);
    expected_output_ = CalculateExpectedResult(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  static InType GenerateLargeMatrix(int size) {
    InType matrix(size, std::vector<int>(size));

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = ((i * 19 + j * 23) % 10000) + 1;
      }
    }

    return matrix;
  }

  static OutType CalculateExpectedResult(const InType &matrix) {
    if (matrix.empty()) {
      return {};
    }

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    OutType result(cols, std::numeric_limits<int>::min());

    for (size_t j = 0; j < cols; ++j) {
      for (size_t i = 0; i < rows; ++i) {
        result[j] = std::max(matrix[i][j], result[j]);
      }
    }

    return result;
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