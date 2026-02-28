#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnilkovEMaxValInColOfMatRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kMatrixSize_ = 100;

  void SetUp() override {
    input_data_ = GenerateLargeMatrix();
    expected_output_ = CalculateExpectedResult(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType expected_output_{};

  InType static GenerateLargeMatrix() {
    return GenerateMatrix(kMatrixSize_, [](int i, int j) { return ((i * 19 + j * 23) % 10000) + 1; });
  }

  InType static GenerateMatrix(int size, auto element_generator) {
    InType matrix(size, std::vector<int>(size));
    FillMatrix(matrix, element_generator);
    return matrix;
  }

  void static FillMatrix(InType &matrix, auto element_generator) {
    for (size_t i = 0; i < matrix.size(); ++i) {
      for (size_t j = 0; j < matrix[i].size(); ++j) {
        matrix[i][j] = element_generator(i, j);
      }
    }
  }

  OutType static CalculateExpectedResult(const InType &matrix) {
    return FindMaxInEachColumn(matrix);
  }

  OutType static FindMaxInEachColumn(const InType &matrix) {
    if (matrix.empty()) {
      return {};
    }
    size_t cols = matrix[0].size();
    OutType result(cols, std::numeric_limits<int>::min());

    for (size_t j = 0; j < cols; ++j) {
      result[j] = FindColumnMax(matrix, j);
    }
    return result;
  }

  int static FindColumnMax(const InType &matrix, size_t col) {
    int max_val = std::numeric_limits<int>::min();
    for (size_t i = 0; i < matrix.size(); ++i) {
      if (matrix[i][col] > max_val) {
        max_val = matrix[i][col];
      }
    }
    return max_val;
  }
};

TEST_P(LuchnilkovEMaxValInColOfMatRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuchnilkovEMaxValInColOfMatMPI, LuchnilkovEMaxValInColOfMatSEQ>(
        PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = LuchnilkovEMaxValInColOfMatRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnilkovEMaxValInColOfMatRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace luchnikov_e_max_val_in_col_of_mat
