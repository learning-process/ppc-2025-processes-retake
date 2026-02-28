#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnilkovEMaxValInColOfMatRunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  LuchnilkovEMaxValInColOfMatRunFuncTestsProcesses() = default;  // ИСПРАВЛЕНО

  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {  // ИСПРАВЛЕНО: убрал static
    auto [matrix_size, test_type] = ExtractTestParams();
    input_data_ = GenerateTestMatrix(matrix_size, test_type);
    expected_output_ = CalculateColumnMaxima(input_data_);
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

  static std::pair<int, std::string> ExtractTestParams() {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    return {std::get<0>(params), std::get<1>(params)};
  }

  static InType GenerateTestMatrix(int size, const std::string &test_type) {
    auto generator = SelectMatrixGenerator(test_type, size);
    return BuildMatrix(size, generator);
  }

  using ElementGenerator = std::function<int(int, int)>;

  static ElementGenerator SelectMatrixGenerator(const std::string &test_type, int size) {
    static const std::unordered_map<std::string, ElementGenerator> kGenerators = {
        {"random1", [](int i, int j) { return (i * 17 + j * 13) % 100; }},
        {"ascending", [size](int i, int j) { return i * size + j + 1; }},
        {"descending", [size](int i, int j) { return size * size - (i * size + j); }},
        {"constant", [](int, int) { return 42; }},
        {"diagonal", [](int i, int j) { return (i == j) ? 1000 : 1; }},
        {"negative", [](int i, int j) { return -(((i * 17 + j * 13) % 100) + 1); }},
        {"mixed", [](int i, int j) { return ((i * 17 + j * 13) % 201) - 100; }},
        {"single_max", [size](int i, int j) { return (i == size / 2 && j == size / 2) ? 10000 : 1; }},
        {"first_col_max", [](int, int j) { return j + 1; }},
        {"last_col_max", [size](int, int j) { return size - j; }},
        {"random2", [](int i, int j) { return ((i + 1) * (j + 1) * 7) % 150; }}};

    auto it = kGenerators.find(test_type);
    if (it == kGenerators.end()) {
      return [](int, int) { return 0; };
    }
    return it->second;
  }

  static InType BuildMatrix(int size, const ElementGenerator &generator) {
    InType matrix(size, std::vector<int>(size));
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = generator(i, j);
      }
    }
    return matrix;
  }

  static OutType CalculateColumnMaxima(const InType &matrix) {
    if (matrix.empty()) {
      return {};
    }
    return FindMaxInColumns(matrix);
  }

  static OutType FindMaxInColumns(const InType &matrix) {
    size_t cols = matrix[0].size();
    OutType result(cols);

    for (size_t j = 0; j < cols; ++j) {
      result[j] = FindColumnMaximum(matrix, j);
    }
    return result;
  }

  static int FindColumnMaximum(const InType &matrix, size_t col) {
    int max_val = std::numeric_limits<int>::min();
    for (const auto &row : matrix) {
      if (row[col] > max_val) {
        max_val = row[col];
      }
    }
    return max_val;
  }
};

namespace {

TEST_P(LuchnilkovEMaxValInColOfMatRunFuncTestsProcesses, MaxValInColTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {std::make_tuple(3, "random1"),       std::make_tuple(5, "ascending"),
                                             std::make_tuple(7, "descending"),    std::make_tuple(4, "constant"),
                                             std::make_tuple(6, "diagonal"),      std::make_tuple(8, "negative"),
                                             std::make_tuple(10, "mixed"),        std::make_tuple(3, "single_max"),
                                             std::make_tuple(5, "first_col_max"), std::make_tuple(7, "last_col_max")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LuchnilkovEMaxValInColOfMatMPI, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat),
                                           ppc::util::AddFuncTask<LuchnilkovEMaxValInColOfMatSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kPerfTestName = LuchnilkovEMaxValInColOfMatRunFuncTestsProcesses::PrintFuncTestName<
    LuchnilkovEMaxValInColOfMatRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixMaxColTests, LuchnilkovEMaxValInColOfMatRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_max_val_in_col_of_mat
