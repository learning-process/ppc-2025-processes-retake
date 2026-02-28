#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

using MatrixGenerator = std::function<void(InType &, int)>;

class LuchnikovEMaxValInColOfMatFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

  LuchnikovEMaxValInColOfMatFuncTests() = default;

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

  static void FillPattern1(InType &matrix, int size) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = (i * 17 + j * 13) % 100;
      }
    }
  }

  static void FillPattern2(InType &matrix, int size) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = (i * size) + j + 1;
      }
    }
  }

  static void FillPattern3(InType &matrix, int size) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = (size * size) - ((i * size) + j);
      }
    }
  }

  static void FillPattern4(InType &matrix, int size) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = 42;
      }
    }
  }

  static void FillPattern5(InType &matrix, int size) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = (i == j) ? 1000 : 1;
      }
    }
  }

  static void FillPattern6(InType &matrix, int size) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = -(((i * 17 + j * 13) % 100) + 1);
      }
    }
  }

  static void FillPattern7(InType &matrix, int size) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        int val = ((i * 17 + j * 13) % 201) - 100;
        matrix[i][j] = val;
      }
    }
  }

  static void FillPattern8(InType &matrix, int size) {
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
  }

  static void FillPattern9(InType &matrix, int size) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = j + 1;
      }
    }
  }

  static void FillPattern10(InType &matrix, int size) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = size - j;
      }
    }
  }

  static void FillPattern11(InType &matrix, int size) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        matrix[i][j] = ((i + 1) * (j + 1) * 7) % 150;
      }
    }
  }

  InType GenerateTestMatrix(int size, const std::string &test_type) {
    InType matrix(size, std::vector<int>(size));

    static const std::unordered_map<std::string, MatrixGenerator> generators = {
        {"pattern1", FillPattern1},   {"pattern2", FillPattern2},  {"pattern3", FillPattern3},
        {"pattern4", FillPattern4},   {"pattern5", FillPattern5},  {"pattern6", FillPattern6},
        {"pattern7", FillPattern7},   {"pattern8", FillPattern8},  {"pattern9", FillPattern9},
        {"pattern10", FillPattern10}, {"pattern11", FillPattern11}};

    auto it = generators.find(test_type);
    if (it != generators.end()) {
      it->second(matrix, size);
    }

    return matrix;
  }

  OutType CalculateExpectedResult(const InType &matrix) {
    if (matrix.empty()) {
      return {};
    }

    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    OutType result(cols, std::numeric_limits<int>::min());

    for (size_t j = 0; j < cols; ++j) {
      for (size_t i = 0; i < rows; ++i) {
        if (matrix[i][j] > result[j]) {
          result[j] = matrix[i][j];
        }
      }
    }

    return result;
  }
};

namespace {

TEST_P(LuchnikovEMaxValInColOfMatFuncTests, MaxValInColumnsTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 11> kTestParam = {
    std::make_tuple(3, "pattern1"),  std::make_tuple(5, "pattern2"), std::make_tuple(7, "pattern3"),
    std::make_tuple(4, "pattern4"),  std::make_tuple(6, "pattern5"), std::make_tuple(8, "pattern6"),
    std::make_tuple(10, "pattern7"), std::make_tuple(3, "pattern8"), std::make_tuple(5, "pattern9"),
    std::make_tuple(7, "pattern10"), std::make_tuple(9, "pattern11")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LuchnikovEMaxValInColOfMatMPI, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat),
                                           ppc::util::AddFuncTask<LuchnikovEMaxValInColOfMatSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LuchnikovEMaxValInColOfMatFuncTests::PrintFuncTestName<LuchnikovEMaxValInColOfMatFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixColumnTests, LuchnikovEMaxValInColOfMatFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace luchnikov_e_max_val_in_col_of_mat
