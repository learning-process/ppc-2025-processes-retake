#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "luchnikov_e_max_val_in_col_of_mat/common/include/common.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/mpi/include/ops_mpi.hpp"
#include "luchnikov_e_max_val_in_col_of_mat/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace luchnikov_e_max_val_in_col_of_mat {

using FullParamType = std::tuple<std::function<std::shared_ptr<BaseTask>(InType)>, std::string, TestType>;

class LuchnikovEMaxValInColOfMatFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, FullParamType> {
 public:
  static std::string PrintTestParam(const testing::TestParamInfo<FullParamType> &info) {
    const auto &params = std::get<2>(info.param);
    return std::to_string(std::get<0>(params)) + "_" + std::get<1>(params);
  }

 protected:
  void SetUp() override {
    const auto &full_params = GetParam();
    const auto &params = std::get<2>(full_params);

    int matrix_size = std::get<0>(params);
    std::string test_type = std::get<1>(params);

    input_data_.resize(matrix_size, std::vector<int>(matrix_size));

    for (int i = 0; i < matrix_size; ++i) {
      for (int j = 0; j < matrix_size; ++j) {
        if (test_type == "type1") {
          input_data_[i][j] = (i + 1) * (j + 1) * 2;
        } else if (test_type == "type2") {
          input_data_[i][j] = (i * 5 + j * 7) % 200;
        } else if (test_type == "type3") {
          input_data_[i][j] = (matrix_size - i) * 5 + j;
        } else if (test_type == "type4") {
          input_data_[i][j] = i * j * 3;
        } else if (test_type == "type5") {
          input_data_[i][j] = (i + j) % 100;
        } else if (test_type == "type6") {
          input_data_[i][j] = (i == j) ? 500 : (i + j + 1);
        } else if (test_type == "type7") {
          input_data_[i][j] = -(i * 3 + j * 2 + 1);
        } else if (test_type == "type8") {
          input_data_[i][j] = ((i + 1) * 11 + (j + 1) * 13) % 300;
        } else if (test_type == "type9") {
          input_data_[i][j] = 200 - i * 4 - j * 3;
        } else if (test_type == "type10") {
          input_data_[i][j] = (i + 1) * (j + 1) * 4;
        }
      }
    }

    expected_output_.resize(matrix_size, INT_MIN);
    for (int j = 0; j < matrix_size; ++j) {
      for (int i = 0; i < matrix_size; ++i) {
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

 private:
  InType input_data_;
  OutType expected_output_;
};

TEST_P(LuchnikovEMaxValInColOfMatFuncTests, MaxInColumnsTest) {
  ExecuteTest(GetParam());
}

namespace {

const std::array<TestType, 10> kTestParam = {std::make_tuple(4, "type1"),  std::make_tuple(5, "type2"),
                                             std::make_tuple(6, "type3"),  std::make_tuple(7, "type4"),
                                             std::make_tuple(8, "type5"),  std::make_tuple(9, "type6"),
                                             std::make_tuple(10, "type7"), std::make_tuple(11, "type8"),
                                             std::make_tuple(12, "type9"), std::make_tuple(13, "type10")};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<LuchnikovEMaxValInColOfMatMPI, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat),
                                           ppc::util::AddFuncTask<LuchnikovEMaxValInColOfMatSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_luchnikov_e_max_val_in_col_of_mat));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(MatrixColumnTests, LuchnikovEMaxValInColOfMatFuncTests, kGtestValues,
                         LuchnikovEMaxValInColOfMatFuncTests::PrintTestParam);

}  // namespace

}  // namespace luchnikov_e_max_val_in_col_of_mat
