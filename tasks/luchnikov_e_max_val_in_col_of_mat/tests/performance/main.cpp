#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnilkovEMaxValInColOfMatRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kMatrixSize = 100;

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
  InType input_data_;
  OutType expected_output_;

  static InType GenerateLargeMatrix() {
    return GenerateMatrix(kMatrixSize, [](int i, int j) { return ((i * 19 + j * 23) % 10000) + 1; });
  }

  template <typename Generator>
  static InType GenerateMatrix(int size, Generator element_generator) {
    InType matrix(size, std::vector<int>(size));
    FillMatrix(matrix, element_generator);
    return matrix;
  }

  template <typename Generator>
  static void FillMatrix(InType &matrix, Generator element_generator) {
    for (size_t i = 0; i < matrix.size(); ++i) {
      for (size_t j = 0; j < matrix[i].size(); ++j) {
        matrix[i][j] = element_generator(static_cast<int>(i), static_cast<int>(j));
      }
    }
  }

  static OutType CalculateExpectedResult(const InType &matrix) {
    return FindMaxInEachColumn(matrix);
  }

  static OutType FindMaxInEachColumn(const InType &matrix) {
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

  static int FindColumnMax(const InType &matrix, size_t col) {
    int max_val = std::numeric_limits<int>::min();
    for (const auto &row : matrix) {
      max_val = std::max(max_val, row[col]);
    }
    return max_val;
  }
};

namespace {

TEST_P(LuchnilkovEMaxValInColOfMatRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, LuchnilkovEMaxValInColOfMatMpi, LuchnilkovEMaxValInColOfMatSeq>(
        PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = LuchnilkovEMaxValInColOfMatRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, LuchnilkovEMaxValInColOfMatRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_max_val_in_col_of_mat
