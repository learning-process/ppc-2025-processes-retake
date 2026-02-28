#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnikovEMaxValInColOfMatFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int matrix_size = std::get<0>(params);
    std::string test_type = std::get<1>(params);

    input_data_ = GenerateTestMatrix(matrix_size, test_type);
    expected_output_ = CalculateExpectedResult(input_data_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;

  static InType GenerateTestMatrix(int size, const std::string &test_type) {
    InType matrix(size, std::vector<int>(size));

    if (test_type == "pattern1") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = (i * 17 + j * 13) % 100;
        }
      }
    } else if (test_type == "pattern2") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = (i * size) + j + 1;
        }
      }
    } else if (test_type == "pattern3") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = (size * size) - ((i * size) + j);
        }
      }
    } else if (test_type == "pattern4") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = 42;
        }
      }
    } else if (test_type == "pattern5") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = (i == j) ? 1000 : 1;
        }
      }
    } else if (test_type == "pattern6") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = -(((i * 17 + j * 13) % 100) + 1);
        }
      }
    } else if (test_type == "pattern7") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          int val = ((i * 17 + j * 13) % 201) - 100;
          matrix[i][j] = val;
        }
      }
    } else if (test_type == "pattern8") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = 1;
        }
      }
      int max_row = size / 2;
      int max_col = size / 2;
      if (size > 0) {
        matrix[max_row][max_col] = 10000;
      }
    } else if (test_type == "pattern9") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = j + 1;
        }
      }
    } else if (test_type == "pattern10") {
      for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
          matrix[i][j] = size - j;
        }
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

namespace {

TEST_P(LuchnikovEMaxValInColOfMatFuncTests, MaxValInColumnsTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {std::make_tuple(3, "pattern1"),  std::make_tuple(5, "pattern2"),
                                             std::make_tuple(7, "pattern3"),  std::make_tuple(4, "pattern4"),
                                             std::make_tuple(6, "pattern5"),  std::make_tuple(8, "pattern6"),
                                             std::make_tuple(10, "pattern7"), std::make_tuple(3, "pattern8"),
                                             std::make_tuple(5, "pattern9"),  std::make_tuple(7, "pattern10")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LuchnikovEMaxValInColOfMatMPI, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat),
                                           ppc::util::AddFuncTask<LuchnikovEMaxValInColOfMatSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LuchnikovEMaxValInColOfMatFuncTests::PrintFuncTestName<LuchnikovEMaxValInColOfMatFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixColumnTests, LuchnikovEMaxValInColOfMatFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_max_val_in_col_of_mat
