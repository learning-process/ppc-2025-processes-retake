#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

class LuchnikovEMaxValInColOfMatFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<TestType> &info) {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(info.param);
    return std::to_string(std::get<0>(params)) + "x" + std::to_string(std::get<1>(params)) + "_" + std::get<2>(params);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int rows = std::get<0>(params);
    int cols = std::get<1>(params);
    std::string matrix_type = std::get<2>(params);

    input_data_ = std::vector<std::vector<int>>(rows, std::vector<int>(cols));

    if (matrix_type == "increasing") {
      int val = 1;
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          input_data_[i][j] = val++;
        }
      }
    } else if (matrix_type == "decreasing") {
      int val = rows * cols;
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          input_data_[i][j] = val--;
        }
      }
    } else if (matrix_type == "random") {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          input_data_[i][j] = (i * cols + j) % 100;
        }
      }
    } else if (matrix_type == "same") {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          input_data_[i][j] = 42;
        }
      }
    } else if (matrix_type == "negative") {
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          input_data_[i][j] = -(i * cols + j + 1);
        }
      }
    }

    expected_output_.resize(cols);
    for (int j = 0; j < cols; ++j) {
      int max_val = input_data_[0][j];
      for (int i = 1; i < rows; ++i) {
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

 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(LuchnikovEMaxValInColOfMatFuncTests, TestMaxInColumns) {
  ExecuteTest(GetParam());
}

namespace {

const std::array<TestType, 10> kTestParams = {std::make_tuple(1, 1, "increasing"), std::make_tuple(2, 2, "increasing"),
                                              std::make_tuple(3, 3, "decreasing"), std::make_tuple(4, 5, "random"),
                                              std::make_tuple(5, 3, "random"),     std::make_tuple(3, 5, "same"),
                                              std::make_tuple(6, 4, "negative"),   std::make_tuple(7, 7, "increasing"),
                                              std::make_tuple(8, 3, "decreasing"), std::make_tuple(4, 8, "random")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LuchnikovEMaxValInColOfMatMPI, InType>(
                                               kTestParams, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat),
                                           ppc::util::AddFuncTask<LuchnikovEMaxValInColOfMatSEQ, InType>(
                                               kTestParams, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(MaxInColumnTests, LuchnikovEMaxValInColOfMatFuncTests, kGtestValues,
                         &LuchnikovEMaxValInColOfMatFuncTests::PrintTestParam);

}  // namespace

}  // namespace luchnikov_e_max_val_in_col_of_mat
