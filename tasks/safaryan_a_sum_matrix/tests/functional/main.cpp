#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "safaryan_a_sum_matrix  /common/include/common.hpp"
#include "safaryan_a_sum_matrix  /mpi/include/ops_mpi.hpp"
#include "safaryan_a_sum_matrix  /seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace safaryan_a_sum_matrix {

class SafaryanASumMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string matrix_name = std::get<1>(test_params);

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_safaryan_a_sum_matrix, matrix_name + ".txt");
    std::ifstream file(abs_path);

    int rows = 0;
    int cols = 0;

    file >> rows >> cols;

    input_data_.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        file >> input_data_[i][j];
      }
    }

    expected_result_.resize(rows);
    for (int i = 0; i < rows; ++i) {
      expected_result_[i] = std::accumulate(input_data_[i].begin(), input_data_[i].end(), 0);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_;
};

namespace {

TEST_P(SafaryanASumMatrixFuncTests, MatrixRowSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(1, "matrix1"), std::make_tuple(2, "matrix2"),
                                            std::make_tuple(3, "matrix3")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SafaryanASumMatrixMPI, InType>(kTestParam, PPC_SETTINGS_safaryan_a_sum_matrix),
    ppc::util::AddFuncTask<SafaryanASumMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_safaryan_a_sum_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SafaryanASumMatrixFuncTests::PrintFuncTestName<SafaryanASumMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatrixRowSum, SafaryanASumMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace safaryan_a_sum_matrix
